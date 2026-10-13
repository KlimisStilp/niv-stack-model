"""Publication-style vector (PDF) figures for the NIV backtest.

All figures are produced from the stored forecasts (``predictions.parquet``) and the stack weights
(``weights.csv``) written by :func:`niv.backtest.save`, so they can be regenerated at any time with
``niv figures`` without refitting anything.

Style: serif text matching the LaTeX body (Computer Modern maths), 8 pt, single-column (3.4 in) or
double-column (6.9 in) widths, embedded TrueType fonts (Type 42), recessive horizontal grid, no top
or right spines, marker + line-style redundancy so the figures survive greyscale printing and
colour-vision deficiency. Colours follow the validated categorical order of the ``dataviz`` palette.

References
----------
* Gneiting, Balabdaoui & Raftery (2007), JRSS B 69(2), 243-268 (probabilistic calibration).
* Gneiting & Raftery (2007), JASA 102(477), 359-378 (scoring rules).
* Tufte (2001), The Visual Display of Quantitative Information (data-ink, small multiples).

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from niv.backtest import load_predictions  # noqa: E402
from niv.metrics import crps_proxy  # noqa: E402
from niv.models import LEVELS  # noqa: E402

INK, INK2, GRID = "#1a1a19", "#52514e", "#e3e2de"
SINGLE, DOUBLE = 3.4, 6.9  # inches

#: model -> (label, colour, linestyle, marker)
STYLE = {
    "climatology": ("Climatology", "#8a8984", (0, (1, 1.5)), "x"),
    "neso_calibrated": ("NESO-calibrated", "#eb6834", "-", "s"),
    "lightgbm": ("LightGBM", "#008300", "-", "^"),
    "ondil_gamlss": ("GAMLSS (JSU)", "#6250d6", "-", "D"),
    "equal_weight": ("Equal-weight average", "#52514e", (0, (4, 2)), "v"),
    "stack_static": ("Static stack", "#1baf7a", (0, (4, 2)), "P"),
    "stack_gated": ("Gated stack", "#2a78d6", "-", "o"),
    "stack_gated+ACI": ("Gated stack + ACI", "#e34948", (0, (5, 1.5, 1, 1.5)), "o"),
}


def _style() -> None:
    """Apply the global rcParams."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _save(fig, out: Path, name: str) -> Path:
    """Save ``fig`` as ``<name>.pdf`` in ``out`` and close it."""
    p = out / f"{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def _line(ax, x, y, key, **kw):
    """Plot one model with its consistent style."""
    lab, col, ls, mk = STYLE[key]
    return ax.plot(x, y, color=col, ls=ls, marker=mk, mec="white", mew=0.4, label=lab, **kw)[0]


def fig_reliability(frame, preds, out: Path) -> Path:
    """Reliability: (a) frequency vs level on the diagonal, (b) the same data as a zoomed deviation.

    Panel (b) magnifies the differences by about an order of magnitude, so a grey band shows the
    sampling uncertainty (+/- 1.96 standard errors, clustered by calendar day to respect serial
    dependence) of the empirical frequency of the gated stack.
    """
    y = frame["y"].values
    day = pd.to_datetime(frame["startTime"].values, utc=True).floor("D")
    keys = ["neso_calibrated", "lightgbm", "ondil_gamlss", "stack_gated", "stack_gated+ACI"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"wspace": 0.28})
    a.plot([0, 1], [0, 1], color=INK2, lw=0.8)
    b.axhline(0, color=INK2, lw=0.8)
    ind_g = pd.DataFrame((y[:, None] <= preds["stack_gated"]).astype(float))
    daily = ind_g.groupby(day.values).mean()
    half = 196 * daily.std().values / np.sqrt(len(daily))  # 1.96 SE in pp
    b.fill_between(
        LEVELS, -half, half, color="#2a78d6", alpha=0.12, lw=0, label=r"95% band, Gated stack"
    )
    for k in keys:
        emp = (y[:, None] <= preds[k]).mean(axis=0)
        _line(a, LEVELS, emp, k)
        _line(b, LEVELS, 100 * (emp - LEVELS), k)
    for ax in (a, b):
        ax.set_xlabel(r"Nominal quantile level $\tau$")
        ax.set_xticks(LEVELS[[1, 3, 4, 5, 7]])
        ax.set_xticklabels(["0.05", "0.25", "0.5", "0.75", "0.95"])
    a.set_ylabel("Empirical frequency")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_aspect("equal")
    a.grid(axis="x", visible=True)
    a.set_title("(a) Reliability diagram", loc="left")
    b.set_ylabel(r"Empirical $-$ nominal (pp)")
    b.set_title("(b) Same data, deviation from diagonal", loc="left")
    h, lab = b.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), columnspacing=1.4)
    return _save(fig, out, "fig_reliability")


def fig_reliability_by_lead(frame, preds, out: Path, diagonal: bool = False) -> Path:
    """Small multiples: one reliability panel per lead time (3 x 4 grid, shared axes).

    With ``diagonal=False`` each panel shows empirical minus nominal frequency (pp) with a shaded
    95 % sampling band for the gated stack (+/- 1.96 day-clustered standard errors); with
    ``diagonal=True`` the classic frequency-vs-level plot is drawn instead.
    """
    y, lead = frame["y"].values, frame["lead_h"].values
    day = pd.to_datetime(frame["startTime"].values, utc=True).floor("D")
    leads = np.sort(np.unique(lead))
    keys = ["neso_calibrated", "lightgbm", "ondil_gamlss", "stack_gated", "stack_gated+ACI"]
    ncol = 4
    nrow = int(np.ceil(len(leads) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(DOUBLE, 1.75 * nrow + 0.6),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.12},
    )
    for ax, h in zip(axes.ravel(), leads, strict=False):
        m = lead == h
        if diagonal:
            ax.plot([0, 1], [0, 1], color=INK2, lw=0.7)
        else:
            ax.axhline(0, color=INK2, lw=0.7)
            ind = pd.DataFrame((y[m][:, None] <= preds["stack_gated"][m]).astype(float))
            daily = ind.groupby(day[m].values).mean()
            half = 196 * daily.std().values / np.sqrt(len(daily))
            ax.fill_between(
                LEVELS,
                -half,
                half,
                color="#2a78d6",
                alpha=0.12,
                lw=0,
                label="95% band, Gated stack",
            )
        for k in keys:
            emp = (y[m][:, None] <= preds[k][m]).mean(axis=0)
            _line(ax, LEVELS, emp if diagonal else 100 * (emp - LEVELS), k, lw=1.0, ms=2.4)
        ax.set_title(f"{h:g} h", loc="left", fontsize=8)
        ax.set_xticks([0.05, 0.5, 0.95])
        ax.set_xticklabels(["0.05", "0.5", "0.95"])
        if diagonal:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
    for ax in axes.ravel()[len(leads) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical frequency" if diagonal else r"Emp. $-$ nom. (pp)")
    for ax in axes[-1]:
        ax.set_xlabel(r"Nominal level $\tau$")
    h_, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045), columnspacing=1.4)
    return _save(
        fig, out, "fig_reliability_by_lead_diag" if diagonal else "fig_reliability_by_lead"
    )


def _pinball_by_level(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean pinball loss at each quantile level, shape ``(len(LEVELS),)``."""
    d = q - y[:, None]
    return np.where(d > 0, (1 - LEVELS) * d, -LEVELS * d).mean(axis=0)


def fig_pinball_by_lead(frame, preds, out: Path, relative: bool = False) -> Path:
    """Small multiples of the mean pinball loss against quantile level, one panel per lead.

    ``relative=False`` plots the loss itself (MWh); ``relative=True`` plots the percentage
    improvement over climatology at each level, which makes the small differences between models
    visible (positive = better than climatology).
    """
    y, lead = frame["y"].values, frame["lead_h"].values
    leads = np.sort(np.unique(lead))
    keys = ["neso_calibrated", "lightgbm", "ondil_gamlss", "stack_gated", "stack_gated+ACI"]
    if not relative:
        keys = ["climatology", *keys]
    ncol = 4
    nrow = int(np.ceil(len(leads) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(DOUBLE, 1.75 * nrow + 0.6),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.12},
    )
    for ax, h in zip(axes.ravel(), leads, strict=False):
        m = lead == h
        ref = _pinball_by_level(y[m], preds["climatology"][m])
        if relative:
            ax.axhline(0, color=INK2, lw=0.7)
        for k in keys:
            pl = _pinball_by_level(y[m], preds[k][m])
            _line(ax, LEVELS, 100 * (1 - pl / ref) if relative else pl, k, lw=1.0, ms=2.4)
        ax.set_title(f"{h:g} h", loc="left", fontsize=8)
        ax.set_xticks([0.05, 0.5, 0.95])
        ax.set_xticklabels(["0.05", "0.5", "0.95"])
    for ax in axes.ravel()[len(leads) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Gain vs climatology (%)" if relative else "Mean pinball loss (MWh)")
    for ax in axes[-1]:
        ax.set_xlabel(r"Quantile level $\tau$")
    h_, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045), columnspacing=1.4)
    return _save(fig, out, "fig_pinball_gain_by_lead" if relative else "fig_pinball_by_lead")


def fig_skill_by_lead(frame, preds, out: Path) -> Path:
    """CRPS skill (%) relative to the NESO-calibrated benchmark for each lead time."""
    y, lead = frame["y"].values, frame["lead_h"].values
    base = crps_proxy(y, preds["neso_calibrated"], LEVELS)
    leads = np.sort(np.unique(lead))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    ax.axhline(0, color=INK2, lw=0.8)
    for k in ["lightgbm", "ondil_gamlss", "equal_weight", "stack_gated", "stack_gated+ACI"]:
        c = crps_proxy(y, preds[k], LEVELS)
        s = [100 * (1 - c[lead == h].mean() / base[lead == h].mean()) for h in leads]
        _line(ax, leads, s, k)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("CRPS skill vs NESO-calibrated (%)")
    ax.set_xticks(leads[::2] if len(leads) > 8 else leads)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.2)
    return _save(fig, out, "fig_skill_by_lead")


def fig_weights(weights: pd.DataFrame, out: Path) -> Path:
    """Mean stack weight of each base learner per test month (stacked bars)."""
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    bottom = np.zeros(len(weights))
    x = np.arange(len(weights))
    for k in weights.columns:
        lab, col, _, _ = STYLE[k]
        ax.bar(
            x,
            weights[k].values,
            bottom=bottom,
            color=col,
            width=0.78,
            label=lab,
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += weights[k].values
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Period(m).strftime("%b\n%y") for m in weights.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean gate weight")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=1.2)
    return _save(fig, out, "fig_stack_weights")


def fig_fan(frame, preds, out: Path, lead_h: float = 12.0, end: str | None = None) -> Path:
    """Seven days of 12 h-ahead predictive intervals (50 % and 90 %) against realised NIV."""
    m = frame["lead_h"].values == lead_h
    t = pd.to_datetime(frame["startTime"].values[m], utc=True)
    q, y = preds["stack_gated+ACI"][m], frame["y"].values[m]
    end_t = pd.Timestamp(end, tz="UTC") if end else t.max() - pd.Timedelta(days=3)
    sel = (t > end_t - pd.Timedelta(days=7)) & (t <= end_t)
    t, q, y = t[sel], q[sel], y[sel]
    lv = list(np.round(LEVELS, 6))
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.5))
    ax.fill_between(
        t,
        q[:, lv.index(0.05)],
        q[:, lv.index(0.95)],
        color="#2a78d6",
        alpha=0.16,
        lw=0,
        label="90% interval",
    )
    ax.fill_between(
        t,
        q[:, lv.index(0.25)],
        q[:, lv.index(0.75)],
        color="#2a78d6",
        alpha=0.34,
        lw=0,
        label="50% interval",
    )
    ax.plot(t, q[:, lv.index(0.5)], color="#2a78d6", lw=1.1, label="Median forecast")
    ax.plot(t, y, color=INK, lw=0.8, label="Realised NIV")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("Net imbalance volume (MWh)")
    ax.set_xlim(t.min(), t.max())
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: matplotlib.dates.num2date(v).strftime("%a %d %b"))
    )
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    return _save(fig, out, "fig_fan_12h")


def fig_rolling_coverage(frame, preds, out: Path) -> Path:
    """Monthly empirical coverage of the 50 % and 90 % central intervals (two panels)."""
    y = frame["y"].values
    month = pd.to_datetime(frame["startTime"].values, utc=True).tz_convert(None).to_period("M")
    lv = list(np.round(LEVELS, 6))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharex=True)
    for ax, (lo, hi, nom) in zip(axes, [(0.25, 0.75, 50), (0.05, 0.95, 90)], strict=True):
        ax.axhline(nom, color=INK2, lw=0.8)
        for k in ["neso_calibrated", "ondil_gamlss", "stack_gated", "stack_gated+ACI"]:
            q = preds[k]
            ins = (y >= q[:, lv.index(lo)]) & (y <= q[:, lv.index(hi)])
            cov = pd.Series(ins).groupby(month).mean() * 100
            _line(ax, np.arange(len(cov)), cov.values, k)
        ax.set_title(f"{nom}% central interval", loc="left")
        ax.set_xticks(np.arange(len(cov)))
        ax.set_xticklabels([p.strftime("%b\n%y") for p in cov.index])
        ax.set_ylabel("Empirical coverage (%)")
        ax.yaxis.set_major_locator(MultipleLocator(5 if nom == 90 else 10))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return _save(fig, out, "fig_monthly_coverage")


def make_figures(pred_path: str | Path, weights_path: str | Path, out: str | Path) -> list[Path]:
    """Create all publication figures from stored forecasts and return the PDF paths."""
    _style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame, preds = load_predictions(pred_path)
    weights = pd.read_csv(weights_path, index_col=0)
    return [
        fig_reliability(frame, preds, out),
        fig_reliability_by_lead(frame, preds, out),
        fig_reliability_by_lead(frame, preds, out, diagonal=True),
        fig_pinball_by_lead(frame, preds, out),
        fig_pinball_by_lead(frame, preds, out, relative=True),
        fig_skill_by_lead(frame, preds, out),
        fig_weights(weights, out),
        fig_fan(frame, preds, out),
        fig_rolling_coverage(frame, preds, out),
    ]
