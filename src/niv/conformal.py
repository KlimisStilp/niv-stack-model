"""Adaptive conformal recalibration of central prediction intervals with delayed feedback.

Method
------
For a central interval ``[lo_i, hi_i]`` with nominal miscoverage ``alpha`` (from the stacked
quantiles), the CQR conformity score is ``E_i = max(lo_i - y_i, y_i - hi_i)`` (Romano, Patterson &
Candes, 2019). The calibrated interval is ``[lo_i - qhat_i, hi_i + qhat_i]`` where ``qhat_i`` is the
``(1 - alpha_i)`` empirical quantile of the most recent ``window`` *known* scores, and the working
level is updated online (Adaptive Conformal Inference, Gibbs & Candes, 2021)::

    alpha_{i+1} = alpha_i + gamma (alpha - err_i),   err_i = 1{y_i outside calibrated interval}

which targets long-run coverage ``1 - alpha`` under arbitrary distribution shift.

**Delayed feedback.** A forecast issued ``lead`` hours before delivery cannot use the outcomes of
targets that have not finished (plus the NIV publication lag). Group ``g`` (one lead time) is
processed in target order with ``delay = ceil((lead + 0.5 + lag)/0.5)`` periods: when forecasting
row ``i`` only outcomes of rows ``<= i - delay`` have been revealed. This keeps the backtest honest.

Groups (here: lead time) are calibrated separately - a Mondrian-style partition (Vovk et al.,
2005) - so coverage holds conditionally on lead.

References
----------
* Romano, Patterson & Candes (2019), NeurIPS (conformalised quantile regression).
* Gibbs & Candes (2021), NeurIPS (adaptive conformal inference).
* Zaffran et al. (2022), ICML (adaptive conformal predictions for time series).
* Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random World (Mondrian CP).

"""

from __future__ import annotations

import math

import numpy as np


def aci_interval(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float,
    delay: int,
    gamma: float = 0.01,
    window: int = 1000,
    min_scores: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run delayed-feedback ACI/CQR over one group, ordered by target time.

    Parameters
    ----------
    y, lo, hi:
        Outcomes and the uncalibrated interval bounds, in chronological target order.
    alpha:
        Target miscoverage (e.g. 0.1 for a 90% interval).
    delay:
        Number of rows after which an outcome is revealed (see module docstring).
    gamma:
        ACI step size.
    window:
        Number of most recent revealed scores used for the empirical quantile.
    min_scores:
        Until this many scores are known the interval is left uncalibrated.

    Returns
    -------
    tuple of numpy.ndarray
        Calibrated ``(lo, hi)`` and the applied score adjustment ``qhat`` per row.

    """
    n = len(y)
    scores = np.empty(n)
    qhat_used = np.zeros(n)
    out_lo, out_hi = lo.copy(), hi.copy()
    a_t = alpha
    buf: list[float] = []
    floor = 1.0 / (window + 1)
    for i in range(n):
        j = i - delay
        if j >= 0:
            s = max(lo[j] - y[j], y[j] - hi[j])
            scores[j] = s
            buf.append(s)
            err = 1.0 if s > qhat_used[j] else 0.0
            a_t = min(max(a_t + gamma * (alpha - err), floor), 1.0)
        if len(buf) >= min_scores:
            recent = np.asarray(buf[-window:])
            level = min(1.0, max(0.0, 1.0 - a_t))
            qh = float(np.quantile(recent, level))
            qhat_used[i] = qh
            out_lo[i] = lo[i] - qh
            out_hi[i] = hi[i] + qh
    return out_lo, out_hi, qhat_used


def delay_periods(lead_h: float, lag_h: float = 1.0) -> int:
    """Return the number of half-hour periods before a target's outcome is known."""
    return int(math.ceil((lead_h + 0.5 + lag_h) / 0.5))


def calibrate_quantiles(
    y: np.ndarray,
    Q: np.ndarray,
    levels: np.ndarray,
    lead_h: np.ndarray,
    order: np.ndarray,
    alphas: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05),
    lag_h: float = 1.0,
    gamma: float = 0.01,
    window: int = 1000,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Calibrate every central interval available on the level grid, separately per lead time.

    Parameters
    ----------
    y, Q, levels, lead_h:
        Outcomes ``(n,)``, stacked quantiles ``(n, J)``, their levels and each row's lead (h).
    order:
        Integer sort key giving chronological target order (e.g. ``startTime`` as int64).
    alphas:
        Miscoverage rates; an interval is calibrated when both ``alpha/2`` and ``1-alpha/2``
        are on the grid.
    lag_h:
        NIV publication lag in hours (see :func:`delay_periods`).
    gamma, window:
        ACI step size and score-window length (see :func:`aci_interval`).

    Returns
    -------
    dict
        ``{alpha: (lo_calibrated, hi_calibrated)}`` aligned with the input rows.

    """
    lv = list(np.round(levels, 6))
    res: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for a in alphas:
        lo_l, hi_l = round(a / 2, 6), round(1 - a / 2, 6)
        if lo_l not in lv or hi_l not in lv:
            continue
        lo_all, hi_all = np.empty(len(y)), np.empty(len(y))
        for lead in np.unique(lead_h):
            idx = np.where(lead_h == lead)[0]
            idx = idx[np.argsort(order[idx], kind="stable")]
            lo_c, hi_c, _ = aci_interval(
                y[idx],
                Q[idx, lv.index(lo_l)],
                Q[idx, lv.index(hi_l)],
                a,
                delay_periods(float(lead), lag_h),
                gamma,
                window,
            )
            lo_all[idx], hi_all[idx] = lo_c, hi_c
        res[a] = (lo_all, hi_all)
    return res
