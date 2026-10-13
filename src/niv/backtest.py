"""Expanding-window, leak-free backtest of the NIV stack.

Protocol
--------
1. Calendar months ``m_0 < m_1 < ...``. For each test month ``m_k`` (``k >= first_test``) every base
   learner is fitted on rows whose *target* period ends before ``start(m_k) - embargo`` and predicts
   month ``m_k``. The predictions are stored as out-of-fold (OOF) base forecasts.
2. The meta-learner (:class:`niv.stack.GatedQuantileStack`) for month ``m_k`` is fitted only on OOF
   predictions of earlier months (again with an embargo), so no information from the future is used
   in weights, gate or scale head (Bergmeir & Benitez, 2012; Cerqueira et al., 2020).
3. Stacked quantiles are recalibrated online per lead time by delayed-feedback ACI/CQR
   (:mod:`niv.conformal`).
4. Everything is scored on the months for which a stack exists, so all candidates are compared on
   identical rows.

References
----------
* Bergmeir & Benitez (2012), Information Sciences 191, 192-213.
* Cerqueira, Torgo & Mozetic (2020), Machine Learning 109, 1997-2028.
* Wolpert (1992), Neural Networks 5(2), 241-259.

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from niv.conformal import calibrate_quantiles
from niv.dataset import feature_columns, load_dataset
from niv.metrics import coverage, crps_proxy, diebold_mariano, interval_table
from niv.models import LEVELS, lead_block, make_models
from niv.stack import GATE_COLS, GatedQuantileStack, equal_weight, gate_matrix

log = logging.getLogger(__name__)

EMBARGO = pd.Timedelta(hours=48)


@dataclass
class BacktestResult:
    """Container of aligned predictions and the evaluation frame."""

    frame: pd.DataFrame  # keys + y + lead_block for evaluated rows
    preds: dict[str, np.ndarray]  # model name -> (n, J)
    weights: pd.DataFrame  # mean gate weights per test month
    cal: dict[float, tuple[np.ndarray, np.ndarray]]  # ACI-calibrated intervals (gated stack)


def _month_index(t: pd.Series) -> pd.Series:
    """Calendar-month label (timezone dropped) of each target time."""
    return pd.Series(t.dt.tz_localize(None).dt.to_period("M").astype(str).values, index=t.index)


def run_backtest(
    root: str | Path,
    quick: bool = False,
    first_test: int = 3,
    stack_start: int = 5,
    max_months: int | None = None,
) -> BacktestResult:
    """Run the sequential backtest and return aligned predictions.

    Parameters
    ----------
    root:
        Data root containing ``features/niv_dataset.parquet``.
    quick:
        Use smaller/faster base learners.
    first_test:
        Index of the first month used as an out-of-fold test month for base learners.
    stack_start:
        Index of the first month for which a stacked forecast is produced.
    max_months:
        Optionally stop after this many months (for debugging).

    """
    df = load_dataset(root).dropna(subset=["y"]).sort_values(["startTime", "lead_h"])
    df = df.reset_index(drop=True)
    feats = feature_columns(df)
    mon = _month_index(df["startTime"])
    months = sorted(mon.unique())
    if max_months:
        months = months[: first_test + max_months]
    names: list[str] = []
    oof: dict[str, list[np.ndarray]] = {}
    oof_idx: list[np.ndarray] = []
    for k in range(first_test, len(months)):
        te_idx = np.where(mon.values == months[k])[0]
        t0 = df.loc[te_idx, "startTime"].min()
        tr_idx = np.where(df["startTime"].values < (t0 - EMBARGO).to_datetime64())[0]
        Xtr, ytr, Xte = df.iloc[tr_idx], df["y"].values[tr_idx], df.iloc[te_idx]
        log.info("fold %s: train=%d test=%d", months[k], len(tr_idx), len(te_idx))
        for m in make_models(feats, quick):
            t = time.time()
            m.fit(Xtr, ytr)
            oof.setdefault(m.name, []).append(m.predict(Xte))
            log.info("  %-16s %.1fs", m.name, time.time() - t)
        oof_idx.append(te_idx)
    names = list(oof)
    idx = np.concatenate(oof_idx)
    P = {n: np.vstack(v) for n, v in oof.items()}
    fold_of = mon.values[idx]
    y = df["y"].values[idx]
    Z = gate_matrix(df.iloc[idx], GATE_COLS)
    tgt = df["startTime"].values[idx]

    stat_q = np.full((len(idx), len(LEVELS)), np.nan)
    gate_q = stat_q.copy()
    wrows = []
    for k in range(stack_start, len(months)):
        te = fold_of == months[k]
        t0 = pd.Timestamp(tgt[te].min())
        tr = tgt < (t0 - EMBARGO).to_datetime64()
        Qtr = [P[n][tr] for n in names]
        Qte = [P[n][te] for n in names]
        st_static = GatedQuantileStack(LEVELS, gate=False, scale_head=False).fit(Qtr, y[tr])
        st_gate = GatedQuantileStack(LEVELS, gate=True, scale_head=True).fit(Qtr, y[tr], Z[tr])
        stat_q[te] = st_static.predict(Qte)
        gate_q[te] = st_gate.predict(Qte, Z[te])
        w = st_gate.weights(Qte, Z[te]).mean(axis=0)
        wrows.append(pd.Series(w, index=names, name=months[k]))
        log.info("stack %s fitted on %d OOF rows", months[k], tr.sum())

    ev = ~np.isnan(gate_q[:, 0])
    frame = df.iloc[idx][["startTime", "issue", "lead_h", "y"]].reset_index(drop=True)[ev]
    frame = frame.reset_index(drop=True)
    frame["block"] = lead_block(frame["lead_h"].values)
    preds = {n: P[n][ev] for n in names}
    preds["equal_weight"] = equal_weight([P[n][ev] for n in names])
    preds["stack_static"] = stat_q[ev]
    preds["stack_gated"] = gate_q[ev]
    order = frame["startTime"].astype("int64").values
    cal = calibrate_quantiles(
        frame["y"].values, preds["stack_gated"], LEVELS, frame["lead_h"].values, order
    )
    return BacktestResult(frame, preds, pd.DataFrame(wrows), cal)


def _splice(q: np.ndarray, cal: dict[float, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Replace outer quantile columns by ACI-calibrated bounds (levels alpha/2 and 1-alpha/2)."""
    out = q.copy()
    lv = list(np.round(LEVELS, 6))
    for a, (lo, hi) in cal.items():
        out[:, lv.index(round(a / 2, 6))] = lo
        out[:, lv.index(round(1 - a / 2, 6))] = hi
    return np.sort(out, axis=1)


def with_aci(res: BacktestResult) -> dict[str, np.ndarray]:
    """Return ``res.preds`` plus the ACI-recalibrated gated stack (``stack_gated+ACI``)."""
    preds = dict(res.preds)
    if "stack_gated+ACI" not in preds:
        preds["stack_gated+ACI"] = _splice(preds["stack_gated"], res.cal)
    return preds


def load_predictions(path: str | Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read ``predictions.parquet`` written by :func:`save` into ``(frame, {model: (n, J)})``."""
    P = pd.read_parquet(path)
    keys = ["startTime", "issue", "lead_h", "y", "block"]
    groups: dict[str, list[str]] = {}
    for c in P.columns:
        if "__q" in c:
            groups.setdefault(c.split("__q")[0], []).append(c)
    out = {n: P[cols].to_numpy(dtype=float) for n, cols in groups.items()}
    return P[keys].copy(), out


def metrics_by_lead(
    frame: pd.DataFrame, preds: dict[str, np.ndarray], models: list[str] | None = None
) -> pd.DataFrame:
    """Per-lead, per-model scores: CRPS proxy, mean pinball, 50/90% coverage and width, skill.

    Skill is ``1 - CRPS / CRPS_ref`` for the climatology and NESO-calibrated references.
    """
    y, lead = frame["y"].values, frame["lead_h"].values
    lv = list(np.round(LEVELS, 6))
    ref = {r: crps_proxy(y, preds[r], LEVELS) for r in ("climatology", "neso_calibrated")}
    rows = []
    for n in models or list(preds):
        q = preds[n]
        c = crps_proxy(y, q, LEVELS)
        pb = np.where(q > y[:, None], (1 - LEVELS) * (q - y[:, None]), LEVELS * (y[:, None] - q))
        for h in np.sort(np.unique(lead)):
            m = lead == h
            r = {
                "model": n,
                "lead_h": float(h),
                "n": int(m.sum()),
                "crps_proxy": c[m].mean(),
                "pinball": pb[m].mean(),
            }
            for lo, hi, nom in ((0.25, 0.75, 50), (0.05, 0.95, 90)):
                a, b = q[m][:, lv.index(lo)], q[m][:, lv.index(hi)]
                r[f"cov{nom}"] = float(((y[m] >= a) & (y[m] <= b)).mean())
                r[f"width{nom}"] = float((b - a).mean())
            r["skill_vs_clim_pct"] = 100 * (1 - c[m].mean() / ref["climatology"][m].mean())
            r["skill_vs_neso_pct"] = 100 * (1 - c[m].mean() / ref["neso_calibrated"][m].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def evaluate(res: BacktestResult) -> dict[str, pd.DataFrame]:
    """Compute summary tables: scores, by-lead scores, coverage, intervals and DM tests."""
    y = res.frame["y"].values
    preds = with_aci(res)
    rows, by_lead, cov_rows, losses = [], [], [], {}
    for n, q in preds.items():
        c = crps_proxy(y, q, LEVELS)
        losses[n] = c
        pb = np.where(q > y[:, None], (1 - LEVELS) * (q - y[:, None]), LEVELS * (y[:, None] - q))
        rows.append({"model": n, "crps_proxy": c.mean(), "pinball_mean": float(pb.mean())})
        for b in np.unique(res.frame["block"]):
            m = res.frame["block"].values == b
            by_lead.append({"model": n, "lead_block": int(b), "crps_proxy": c[m].mean()})
        cov_rows.append(coverage(y, q, LEVELS).rename(n))
    scores = pd.DataFrame(rows).set_index("model")
    scores["skill_vs_neso"] = 1 - scores["crps_proxy"] / scores.loc["neso_calibrated", "crps_proxy"]
    lead_tab = pd.DataFrame(by_lead).pivot(index="model", columns="lead_block", values="crps_proxy")
    cov = pd.concat(cov_rows, axis=1).T
    intervals = {n: interval_table(y, q, LEVELS) for n, q in preds.items()}
    interval_tab = pd.concat(intervals, names=["model"]).reset_index(level=0)
    period = res.frame["startTime"].astype("int64").values
    per = {n: pd.Series(loss).groupby(period).mean().sort_index() for n, loss in losses.items()}
    dm = []
    for a, b in [
        ("stack_gated", "neso_calibrated"),
        ("stack_gated", "equal_weight"),
        ("stack_gated", "stack_static"),
        ("stack_gated+ACI", "stack_gated"),
        ("ondil_gamlss", "lightgbm"),
    ]:
        s, p = diebold_mariano(per[a].values, per[b].values)
        dm.append({"A": a, "B": b, "dm_stat": s, "p_value": p})
    return {
        "scores": scores,
        "by_lead": lead_tab,
        "coverage": cov,
        "intervals": interval_tab,
        "dm": pd.DataFrame(dm),
        "weights": res.weights,
    }


def save(res: BacktestResult, tables: dict[str, pd.DataFrame], out: str | Path) -> Path:
    """Write predictions (parquet) and tables (CSV + markdown report) to ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    P = res.frame.copy()
    cols = {}
    for n, q in with_aci(res).items():
        for j, a in enumerate(LEVELS):
            cols[f"{n}__q{a:g}"] = np.round(q[:, j], 1).astype("float32")  # 0.1 MWh
    P = pd.concat([P, pd.DataFrame(cols)], axis=1)
    P.to_parquet(out / "predictions.parquet", compression="zstd", compression_level=19)
    md = ["# NIV stack backtest\n"]
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv")
        md.append(f"## {name}\n\n{t.round(4).to_markdown()}\n")
    (out / "report.md").write_text("\n".join(md))
    return out
