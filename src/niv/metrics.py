"""Evaluation of probabilistic (quantile) forecasts.

Metrics
-------
* Pinball loss per level and its integral, a truncated CRPS (:func:`crps_proxy`).
* Empirical coverage per level and for central intervals.
* Winkler / interval score of central intervals.
* Diebold-Mariano test with a Newey-West (HAC) variance for serially correlated loss differences.

References
----------
* Gneiting & Raftery (2007), JASA 102(477), 359-378 (strictly proper scoring rules).
* Gneiting (2011), Int. J. Forecasting 27(2), 197-207 (quantiles and pinball loss).
* Laio & Tamea (2007), HESS 11, 1267-1277 (CRPS as integral of pinball loss).
* Diebold & Mariano (1995), JBES 13(3), 253-263.
* Newey & West (1987), Econometrica 55(3), 703-708.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    """Enforce non-crossing quantiles by sorting each row (monotone rearrangement).

    Chernozhukov, Fernandez-Val & Galichon (2010), Econometrica 78(3), 1093-1125.
    """
    return np.sort(q, axis=1)


def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Pinball loss per observation and level, shape ``(n, len(levels))``.

    ``L_a(q, y) = (1{y < q} - a)(q - y)``.
    """
    diff = q - y[:, None]
    return np.where(diff > 0, (1 - levels[None, :]) * diff, -levels[None, :] * diff)


def crps_proxy(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return the truncated CRPS per observation (2 * integral of pinball over the level range).

    Computed with the trapezoid rule on the supplied (non-uniform) level grid; tails beyond the
    outer levels are excluded, so values are comparable across models but not equal to the CRPS.
    """
    return 2.0 * np.trapezoid(pinball(y, q, levels), levels, axis=1)


def coverage(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> pd.Series:
    """Empirical frequency of ``y <= q_level`` for each level (should equal the level)."""
    return pd.Series((y[:, None] <= q).mean(axis=0), index=levels, name="coverage")


def interval_table(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> pd.DataFrame:
    """Coverage, mean width and Winkler score for central intervals available on the grid."""
    rows = []
    lv = list(np.round(levels, 6))
    for lo in lv:
        hi = round(1 - lo, 6)
        if lo < 0.5 and hi in lv:
            ql, qh = q[:, lv.index(lo)], q[:, lv.index(hi)]
            a = 2 * lo
            inside = (y >= ql) & (y <= qh)
            winkler = (qh - ql) + (2 / a) * (np.maximum(ql - y, 0) + np.maximum(y - qh, 0))
            rows.append(
                {
                    "nominal": 1 - a,
                    "coverage": inside.mean(),
                    "width": (qh - ql).mean(),
                    "winkler": winkler.mean(),
                }
            )
    return pd.DataFrame(rows).sort_values("nominal").reset_index(drop=True)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 48) -> tuple[float, float]:
    """Two-sided DM test of equal expected loss (``A`` vs ``B``) with Newey-West variance.

    Parameters
    ----------
    loss_a, loss_b:
        Per-period losses (e.g. mean CRPS proxy per target period).
    lag:
        Bartlett-kernel truncation lag (48 = one day of half-hourly periods).

    Returns
    -------
    tuple[float, float]
        ``(statistic, p_value)``; a negative statistic favours model A.

    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[~np.isnan(d)]
    n = len(d)
    dm = d - d.mean()
    var = dm @ dm / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (dm[k:] @ dm[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))
