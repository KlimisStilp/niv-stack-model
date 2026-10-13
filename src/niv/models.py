"""Level-0 base learners producing predictive quantiles of NIV.

Every learner implements :class:`QuantileModel`: ``fit(X, y)`` on a feature DataFrame (columns from
:mod:`niv.dataset`) and ``predict(X)`` returning an array ``(n_rows, len(LEVELS))`` of quantiles,
sorted to be non-crossing (:func:`niv.metrics.sort_quantiles`; Chernozhukov et al., 2010).

Learners
--------
* :class:`Climatology`      - empirical quantiles by (local half-hour, weekend flag).
* :class:`NESOCalibrated`   - NESO indicated imbalance, linearly calibrated per lead block, plus
  empirical residual quantiles. A strong, cheap benchmark that also serves as a stack member.
* :class:`LightGBMQuantile` - one gradient-boosted tree model per quantile level (pinball loss).
* :class:`OndilGAMLSS`      - distributional regression (GAMLSS) with parameters of a Johnson-SU
  (or other) distribution as linear functions of features, fitted with the ``ondil`` package.

On GAMLSS in Python
-------------------
GAMLSS (Rigby & Stasinopoulos, 2005) exists natively only in R. The ``ondil`` package (Hirsch,
Berrisch & Ziel, "Online Distributional Regression", arXiv:2407.08750; GPL-3.0) provides an online,
regularisable GAMLSS-type estimator in Python; its ``OnlineDistributionalRegression`` is used here.
Smooth (spline) effects are approximated by the engineered cyclic/linear terms of the dataset;
adding basis expansions is a straightforward extension.

References
----------
* ondil: https://github.com/simon-hirsch/ondil
* Rigby & Stasinopoulos (2005), JRSS C 54(3), 507-554.
* Koenker & Bassett (1978), Econometrica 46(1), 33-50 (quantile loss).
* Ke et al. (2017), LightGBM, NeurIPS.
* Johnson (1949), Biometrika 36, 149-176 (S_U distribution).

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from niv.metrics import sort_quantiles

log = logging.getLogger(__name__)

#: Quantile levels at which all models predict.
LEVELS = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])
LEAD_BLOCKS = (0, 6, 16, 28, 41)  # [0,6), [6,16), [16,28), [28,41)

#: Columns used by the GAMLSS location (mu) equation.
GAMLSS_MU_COLS = [
    "ind_imb",
    "ind_imb_x_lead",
    "ind_missing",
    "ind_imb_rev3h",
    "ind_gap",
    "ndf",
    "emb_proxy",
    "wind_fc",
    "da_solar",
    "resload",
    "margin_surplus",
    "niv_last",
    "niv_m48",
    "niv_m336",
    "niv_seas336",
    "ssp_last",
    "mid_last",
    "fuel_wind",
    "fuel_ccgt",
    "fuel_ic_net",
    "lead_h",
    "pod_sin",
    "pod_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_wkend",
]
#: Columns used by the GAMLSS scale (sigma) equation (volatility / uncertainty drivers).
GAMLSS_SIGMA_COLS = [
    "lead_h",
    "niv_s48",
    "niv_s336",
    "abs_ind_imb",
    "ind_missing",
    "ind_imb_age_h",
    "wind_fc",
    "resload",
    "margin_surplus",
    "pod_sin",
    "pod_cos",
    "ssp_s48",
    "is_wkend",
]


def lead_block(lead_h: np.ndarray) -> np.ndarray:
    """Map lead times (h) to block ids 0..3 using :data:`LEAD_BLOCKS`."""
    return np.clip(np.digitize(lead_h, LEAD_BLOCKS[1:-1]), 0, len(LEAD_BLOCKS) - 2)


class QuantileModel(ABC):
    """Interface of a base learner."""

    name: str = "model"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> QuantileModel:
        """Fit on features ``X`` and labels ``y``; return ``self``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return non-crossing quantiles, shape ``(len(X), len(LEVELS))``."""


class Climatology(QuantileModel):
    """Empirical quantiles of ``y`` by local half-hour of day and weekend flag."""

    name = "climatology"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Climatology:
        """Store quantiles per ``(pod, is_wkend)`` cell and a global fallback."""
        df = pd.DataFrame({"pod": X["pod"].values, "wk": X["is_wkend"].values, "y": y})
        self._global = np.quantile(y, LEVELS)
        self._table = {
            k: np.quantile(g["y"].values, LEVELS)
            for k, g in df.groupby(["pod", "wk"])
            if len(g) >= 50
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Look up the cell quantiles for each row."""
        keys = zip(X["pod"].values, X["is_wkend"].values, strict=True)
        return np.vstack([self._table.get(k, self._global) for k in keys])


class NESOCalibrated(QuantileModel):
    """NESO indicated imbalance, calibrated per lead block, plus empirical residual quantiles.

    ``y ~ a_b + b_b * ind_imb`` is fitted by OLS in each lead block ``b`` (the sign/scale of the
    indicated imbalance differs from NIV: it is a MW quantity with opposite sign convention).
    Predictive quantiles are the point forecast plus empirical quantiles of the block's residuals.
    Rows without an indicated imbalance fall back to :class:`Climatology`.
    """

    name = "neso_calibrated"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> NESOCalibrated:
        """Fit per-block OLS and residual quantiles."""
        self._clim = Climatology().fit(X, y)
        blk = lead_block(X["lead_h"].values)
        x = X["ind_imb"].values
        self._coef, self._resq = {}, {}
        for b in np.unique(blk):
            m = (blk == b) & ~np.isnan(x)
            if m.sum() < 200:
                continue
            A = np.column_stack([np.ones(m.sum()), x[m]])
            coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            self._coef[b] = coef
            self._resq[b] = np.quantile(y[m] - A @ coef, LEVELS)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecast plus block residual quantiles; climatology where unavailable."""
        out = self._clim.predict(X)
        blk = lead_block(X["lead_h"].values)
        x = X["ind_imb"].values
        for b, coef in self._coef.items():
            m = (blk == b) & ~np.isnan(x)
            out[m] = (coef[0] + coef[1] * x[m])[:, None] + self._resq[b][None, :]
        return sort_quantiles(out)


class LightGBMQuantile(QuantileModel):
    """Gradient-boosted trees with the pinball loss, one model per quantile level.

    Handles missing values natively, which matters here because many forecasts are not yet
    published at long lead times.
    """

    name = "lightgbm"

    def __init__(self, features: list[str], max_rows: int = 150_000, **params) -> None:
        """Set feature columns, a training-row cap (evenly strided in time) and LightGBM params."""
        self.features = features
        self.max_rows = max_rows
        self.params = {
            "n_estimators": 200,
            "learning_rate": 0.06,
            "num_leaves": 24,
            "min_child_samples": 300,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "max_bin": 63,
            "n_jobs": 2,
            "verbose": -1,
        }
        self.params.update(params)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LightGBMQuantile:
        """Fit one ``LGBMRegressor(objective='quantile')`` per level."""
        import lightgbm as lgb

        step = max(1, len(X) // self.max_rows)
        Xs = X[self.features].iloc[::step].astype(float)
        ys = y[::step]
        self._models = [
            lgb.LGBMRegressor(objective="quantile", alpha=float(a), **self.params).fit(Xs, ys)
            for a in LEVELS
        ]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict every level and sort to avoid crossing."""
        Xf = X[self.features].astype(float)
        return sort_quantiles(np.column_stack([m.predict(Xf) for m in self._models]))


class OndilGAMLSS(QuantileModel):
    """GAMLSS-type distributional regression using ``ondil.OnlineDistributionalRegression``.

    Location (mu) and scale (sigma) are linear in feature subsets (engineered cyclic terms stand
    in for smooths); shape parameters (nu, tau) are intercept-only, as is common for stable fits.
    Inputs are winsorised at the 0.5/99.5 training percentiles; missing values are filled with 0
    for the indicated-imbalance block (an explicit ``ind_missing`` flag carries the information)
    and with training medians elsewhere.

    If the Johnson-SU fit fails, the Student-t is tried before raising.
    """

    name = "ondil_gamlss"

    def __init__(
        self, distribution: str = "JSU", max_rows: int = 120_000, method: str = "ols"
    ) -> None:
        """Choose the response distribution (an ``ondil.distributions`` class name) and size cap."""
        self.distribution = distribution
        self.max_rows = max_rows
        self.method = method
        self._cols = list(dict.fromkeys(GAMLSS_MU_COLS + GAMLSS_SIGMA_COLS))

    def _prep(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Impute, winsorise and return a float matrix of ``self._cols``."""
        M = X[self._cols].astype(float).copy()
        if fit:
            self._med = M.median()
            self._lo = M.quantile(0.005)
            self._hi = M.quantile(0.995)
        ind_cols = [
            c
            for c in M.columns
            if c.startswith(("ind_", "abs_ind")) and c not in ("ind_missing", "ind_imb_age_h")
        ]
        M[ind_cols] = M[ind_cols].fillna(0.0)
        M = M.fillna(self._med.fillna(0.0)).clip(self._lo, self._hi, axis=1)
        return M.to_numpy()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> OndilGAMLSS:
        """Fit the distributional regression (falls back to StudentT if JSU fails)."""
        from ondil import distributions as D
        from ondil.estimators import OnlineDistributionalRegression

        step = max(1, len(X) // self.max_rows)
        A = self._prep(X.iloc[::step], fit=True)
        ys = np.asarray(y[::step], dtype=float)
        mu_idx = np.array([self._cols.index(c) for c in GAMLSS_MU_COLS])
        sg_idx = np.array([self._cols.index(c) for c in GAMLSS_SIGMA_COLS])
        last_err = None
        for dist_name in dict.fromkeys([self.distribution, "StudentT"]):
            dist = getattr(D, dist_name)()
            eq = {0: mu_idx, 1: sg_idx}
            eq.update({p: "intercept" for p in range(2, dist.n_params)})
            try:
                self._model = OnlineDistributionalRegression(
                    distribution=dist, equation=eq, method=self.method, ic="aic", scale_inputs=True
                ).fit(A, ys)
                self.fitted_distribution = dist_name
                return self
            except Exception as exc:  # noqa: BLE001 - try the next distribution
                log.warning("ondil %s failed: %s", dist_name, exc)
                last_err = exc
        raise RuntimeError("ondil GAMLSS failed for all distributions") from last_err

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the quantile levels from the fitted conditional distribution."""
        q = self._model.predict_quantile(self._prep(X), LEVELS)
        return sort_quantiles(np.asarray(q))


def make_models(features: list[str], quick: bool = False) -> list[QuantileModel]:
    """Return the default base-learner set (smaller/faster when ``quick``)."""
    lgb_kw = {"n_estimators": 60, "max_rows": 40_000} if quick else {}
    return [
        Climatology(),
        NESOCalibrated(),
        LightGBMQuantile(features, **lgb_kw),
        OndilGAMLSS(max_rows=30_000 if quick else 120_000),
    ]
