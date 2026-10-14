"""Tests for the dataset builder primitives, metrics, stack, conformal layer and base learners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from niv.conformal import aci_interval, delay_periods
from niv.dataset import asof_latest, lagged_features
from niv.metrics import crps_proxy, diebold_mariano, pinball, sort_quantiles
from niv.models import LEVELS, Climatology, NESOCalibrated, lead_block
from niv.stack import GatedQuantileStack


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_asof_never_uses_future_publication():
    left = pd.DataFrame(
        {
            "startTime": [_utc("2026-01-02 12:00")] * 2,
            "issue": [_utc("2026-01-02 00:00"), _utc("2026-01-02 06:00")],
        }
    )
    right = pd.DataFrame(
        {
            "startTime": [_utc("2026-01-02 12:00")] * 3,
            "publishTime": [
                _utc("2026-01-01 20:00"),
                _utc("2026-01-02 03:00"),
                _utc("2026-01-02 09:00"),
            ],
            "v": [1.0, 2.0, 3.0],
        }
    )
    out = asof_latest(left, right, ["v"], "v")
    assert out["v"].tolist() == [1.0, 2.0]  # the 09:00 publication is never visible
    assert (out["v_age_h"] >= 0).all()


def test_asof_missing_when_nothing_published():
    left = pd.DataFrame(
        {"startTime": [_utc("2026-01-02 12:00")], "issue": [_utc("2026-01-01 00:00")]}
    )
    right = pd.DataFrame(
        {
            "startTime": [_utc("2026-01-02 12:00")],
            "publishTime": [_utc("2026-01-01 10:00")],
            "v": [5.0],
        }
    )
    assert np.isnan(asof_latest(left, right, ["v"], "v")["v"].iloc[0])


def test_lagged_features_respect_publication_lag():
    idx = pd.date_range("2026-01-01", periods=400, freq="30min", tz="UTC")
    s = pd.Series(np.arange(400, dtype=float), index=idx)
    issue = pd.Series([idx[200]])
    start = pd.Series([idx[210]])
    f = lagged_features(start, issue, s, "x", lag_h=1.0)
    # period k ends at idx[k]+30min and is known 1h later: need idx[k] + 1.5h <= idx[200]
    assert f["x_last"].iloc[0] == 197.0


def test_pinball_known_values():
    y = np.array([0.0])
    q = np.array([[1.0, -1.0]])
    lv = np.array([0.1, 0.9])
    # q=1>y: (1-0.1)*1 = 0.9; q=-1<y: 0.9*1 = 0.9
    assert np.allclose(pinball(y, q, lv), [[0.9, 0.9]])


def test_crps_proxy_matches_integral_for_perfect_grid():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    from scipy.stats import norm

    q = np.tile(norm.ppf(LEVELS), (len(y), 1))
    assert crps_proxy(y, q, LEVELS).mean() < crps_proxy(y, q * 2.5, LEVELS).mean()


def test_sort_quantiles_non_crossing():
    q = np.array([[3.0, 1.0, 2.0]])
    assert np.all(np.diff(sort_quantiles(q), axis=1) >= 0)


def test_diebold_mariano_detects_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 1.0, 3000)
    b = rng.normal(1.3, 1.0, 3000)
    stat, p = diebold_mariano(a, b)
    assert stat < 0 and p < 0.01


def _toy_stack_data(n=600, k=3):
    rng = np.random.default_rng(2)
    y = rng.normal(size=n)
    Z = rng.normal(size=(n, 4))
    base = [
        np.sort(y[:, None] * 0 + rng.normal(scale=s, size=(n, len(LEVELS))), axis=1)
        for s in (0.8, 1.0, 1.5)
    ][:k]
    return np.stack(base, axis=1), y, Z


def test_stack_gradient_matches_finite_differences():
    Qs, y, Z = _toy_stack_data()
    st = GatedQuantileStack(LEVELS).fit(Qs, y, Z)
    Phi = st._design(Z, len(y))
    th = np.random.default_rng(3).normal(0, 0.1, len(st.theta_))
    err = check_grad(
        lambda t: st._loss_grad(t, Phi, Qs, y)[0], lambda t: st._loss_grad(t, Phi, Qs, y)[1], th
    )
    assert err < 1e-5


def test_stack_output_is_monotone_and_weights_sum_to_one():
    Qs, y, Z = _toy_stack_data()
    st = GatedQuantileStack(LEVELS).fit(Qs, y, Z)
    assert np.all(np.diff(st.predict(Qs, Z), axis=1) >= 0)
    assert np.allclose(st.weights(Qs, Z).sum(axis=1), 1.0)


def test_aci_restores_coverage_after_variance_shift():
    rng = np.random.default_rng(4)
    n = 6000
    y = np.concatenate([rng.normal(size=n // 2), rng.normal(scale=3, size=n // 2)])
    lo, hi = np.full(n, -1.645), np.full(n, 1.645)  # 90% interval of the first regime
    clo, chi, _ = aci_interval(y, lo, hi, 0.1, delay=2, gamma=0.02, window=500)
    raw = ((y >= lo) & (y <= hi))[n // 2 + 500 :].mean()
    cal = ((y >= clo) & (y <= chi))[n // 2 + 500 :].mean()
    assert raw < 0.6 and cal > 0.85


def test_aci_uses_only_revealed_outcomes():
    n, delay = 300, 10
    rng = np.random.default_rng(5)
    y = rng.normal(size=n)
    lo, hi = np.full(n, -1.0), np.full(n, 1.0)
    base = aci_interval(y, lo, hi, 0.1, delay)[0]
    y2 = y.copy()
    y2[200:] += 100.0  # change outcomes the forecaster cannot yet have seen at row 205
    alt = aci_interval(y2, lo, hi, 0.1, delay)[0]
    assert np.allclose(base[: 200 + delay], alt[: 200 + delay])


def test_delay_periods():
    assert delay_periods(2.0, 1.0) == 7


def _frame(n=3000):
    rng = np.random.default_rng(6)
    return pd.DataFrame(
        {
            "pod": rng.integers(1, 49, n),
            "is_wkend": rng.integers(0, 2, n),
            "lead_h": rng.choice([2, 12, 30], n),
            "ind_imb": rng.normal(size=n),
        }
    )


def test_base_learners_shape_and_monotone():
    X = _frame()
    y = 2 * X["ind_imb"].values + np.random.default_rng(7).normal(size=len(X))
    for m in (Climatology(), NESOCalibrated()):
        q = m.fit(X, y).predict(X)
        assert q.shape == (len(X), len(LEVELS)) and np.all(np.diff(q, axis=1) >= 0)
    # the calibrated indicated imbalance should beat climatology in-sample
    c = crps_proxy(y, Climatology().fit(X, y).predict(X), LEVELS).mean()
    n = crps_proxy(y, NESOCalibrated().fit(X, y).predict(X), LEVELS).mean()
    assert n < c


def test_lead_block_range():
    assert set(lead_block(np.array([2, 6, 16, 28, 40]))) == {0, 1, 2, 3}


@pytest.mark.slow
def test_ondil_gamlss_runs_on_synthetic_data():
    pytest.importorskip("ondil")
    from niv.models import GAMLSS_MU_COLS, GAMLSS_SIGMA_COLS, OndilGAMLSS

    cols = list(dict.fromkeys(GAMLSS_MU_COLS + GAMLSS_SIGMA_COLS))
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(3000, len(cols))), columns=cols)
    y = X["niv_last"].values + rng.normal(size=3000) * np.exp(0.3 * X["niv_s48"].values)
    q = OndilGAMLSS(max_rows=3000).fit(X, y).predict(X)
    assert q.shape == (3000, len(LEVELS)) and np.isfinite(q).all()
    assert abs((y <= q[:, 4]).mean() - 0.5) < 0.05


def test_figures_pipeline_writes_pdfs(tmp_path):
    from niv.backtest import BacktestResult, load_predictions, save
    from niv.reports import make_figures

    rng = np.random.default_rng(9)
    n = 12 * 48 * 30
    t = pd.date_range("2026-03-01", periods=n // 12, freq="30min", tz="UTC").repeat(12)
    lead = np.tile(np.arange(2, 26, 2), n // 12)
    y = rng.normal(0, 300, n)
    base = np.sort(rng.normal(0, 300, (n, len(LEVELS))), axis=1)
    names = [
        "climatology",
        "neso_calibrated",
        "lightgbm",
        "ondil_gamlss",
        "equal_weight",
        "stack_static",
        "stack_gated",
    ]
    preds = {k: base + i for i, k in enumerate(names)}
    frame = pd.DataFrame(
        {
            "startTime": t,
            "issue": t - pd.to_timedelta(lead, unit="h"),
            "lead_h": lead.astype(float),
            "y": y,
            "block": 0,
        }
    )
    w = pd.DataFrame(0.25, index=["2026-03"], columns=names[:4])
    cal = {0.1: (preds["stack_gated"][:, 1], preds["stack_gated"][:, 7])}
    save(BacktestResult(frame, preds, w, cal), {}, tmp_path)
    f2, p2 = load_predictions(tmp_path / "predictions.parquet")
    assert "stack_gated+ACI" in p2 and p2["stack_gated"].shape == (n, len(LEVELS))
    w.to_csv(tmp_path / "weights.csv")
    pdfs = make_figures(tmp_path / "predictions.parquet", tmp_path / "weights.csv", tmp_path / "f")
    assert len(pdfs) == 9 and all(p.stat().st_size > 1000 for p in pdfs)


def test_weather_leak_rule():
    from niv.dataset import wx_day_for_lead

    assert [wx_day_for_lead(h) for h in (2, 16, 17, 40)] == [1, 1, 2, 2]
