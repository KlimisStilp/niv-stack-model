r"""Level-1 meta-learner: regime-aware quantile stacking.

Model (methodology Section 5.2, eqs. (qavg), (gate))
----------------------------------------------------
For row ``i`` with base-learner quantile vectors ``Q_k[i, :]`` (k = 1..K) and gate features
``z_i``::

    w_ik   = softmax_k( B_k . phi(z_i) )                     weights, phi(z) = [1, z]
    Qhat_i = sum_k w_ik Q_k[i, :]                            quantile averaging (Vincentisation)
    m_i    = Qhat_i[median level]
    s_i    = exp( gamma . psi(z_i) )                         optional scale head
    Qtil_i = m_i + s_i (Qhat_i - m_i)                        regime-dependent dispersion correction

``B`` (and ``gamma``) minimise the mean pinball loss (a CRPS proxy) with a ridge penalty, using
L-BFGS and the analytic gradient implemented below (verified against finite differences in the
tests). With ``gate=False`` the weights are constant (a static stack); with ``scale_head=False``
``s_i = 1``. Quantile averaging is monotone, so the output is non-crossing whenever the inputs are
and ``s_i > 0``.

References
----------
* Wolpert (1992), Neural Networks 5(2), 241-259 (stacked generalisation).
* Breiman (1996), Machine Learning 24, 49-64 (non-negative stacking weights).
* Jacobs et al. (1991), Neural Computation 3(1), 79-87 (gating / mixture of experts).
* Lichtendahl, Grushka-Cockayne & Winkler (2013), Management Science 59(7), 1594-1611
  (averaging quantiles vs probabilities).
* Sill et al. (2009), arXiv:0911.0460 (feature-weighted linear stacking).
* Gneiting & Ranjan (2013), EJS 7, 1747-1782 (combining predictive distributions).

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from niv.metrics import sort_quantiles


class GatedQuantileStack:
    """Softmax-gated quantile-averaging stack with optional scale head."""

    def __init__(
        self,
        levels: np.ndarray,
        gate: bool = True,
        scale_head: bool = True,
        l2: float = 1e-2,
        max_iter: int = 300,
    ) -> None:
        """Configure the stack.

        Parameters
        ----------
        levels:
            Quantile levels of the base-learner predictions (columns of each ``Q_k``).
        gate:
            Let the weights depend on gate features (else constant weights).
        scale_head:
            Learn the regime-dependent dispersion multiplier ``s``.
        l2:
            Ridge penalty on non-intercept coefficients (loss is normalised to O(1)).
        max_iter:
            L-BFGS iterations.

        """
        self.levels = np.asarray(levels, dtype=float)
        self.gate, self.scale_head, self.l2, self.max_iter = gate, scale_head, l2, max_iter
        self._med = int(np.argmin(np.abs(self.levels - 0.5)))

    # -- gate features --------------------------------------------------------------------------
    def _design(self, Z: np.ndarray | None, n: int) -> np.ndarray:
        """Return ``[1, standardised Z]`` (just ``1`` when the gate is off)."""
        if not self.gate or Z is None:
            return np.ones((n, 1))
        Zs = (np.nan_to_num(Z, nan=0.0) - self._zmean) / self._zstd
        return np.column_stack([np.ones(n), np.clip(Zs, -5, 5)])

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split the flat parameter vector into ``B (K, d)`` and ``gamma (d_s,)``."""
        B = theta[: self.K * self.d].reshape(self.K, self.d)
        return B, theta[self.K * self.d :]

    def _forward(self, theta: np.ndarray, Phi: np.ndarray, Qs: np.ndarray) -> tuple:
        """Compute weights, combined quantiles ``Qhat``, scale ``s`` and output ``Qtil``."""
        B, gamma = self._unpack(theta)
        W = softmax(Phi @ B.T, axis=1)  # (n, K)
        Qhat = np.einsum("nk,nkj->nj", W, Qs)
        if self.scale_head:
            s = np.exp(np.clip(Phi @ gamma, -2.0, 2.0))
        else:
            s = np.ones(len(Phi))
        m = Qhat[:, self._med : self._med + 1]
        Qtil = m + s[:, None] * (Qhat - m)
        return W, Qhat, s, Qtil

    def _loss_grad(self, theta, Phi, Qs, y):
        """Penalised mean pinball loss (normalised by ``self._norm``) and its gradient."""
        n, J = len(y), len(self.levels)
        B, gamma = self._unpack(theta)
        W, Qhat, s, Qtil = self._forward(theta, Phi, Qs)
        diff = Qtil - y[:, None]
        pl = np.where(diff > 0, (1 - self.levels) * diff, -self.levels * diff)
        loss = pl.mean() / self._norm
        g = (np.where(diff > 0, 1.0, 0.0) - self.levels[None, :]) / (n * J * self._norm)  # dL/dQtil
        m = Qhat[:, self._med]
        dQhat = s[:, None] * g
        dQhat[:, self._med] += (1 - s) * g.sum(axis=1)
        dW = np.einsum("nj,nkj->nk", dQhat, Qs)
        dlogit = W * (dW - (W * dW).sum(axis=1, keepdims=True))
        gB = dlogit.T @ Phi
        if self.scale_head:
            ds = (g * (Qhat - m[:, None])).sum(axis=1)
            active = (np.abs(Phi @ gamma) < 2.0).astype(float)  # clip has zero gradient outside
            ggamma = Phi.T @ (ds * s * active)
        else:
            ggamma = np.zeros_like(gamma)
        pen = self.l2 * ((B[:, 1:] ** 2).sum() + (gamma[1:] ** 2).sum())
        gB[:, 1:] += 2 * self.l2 * B[:, 1:]
        ggamma[1:] += 2 * self.l2 * gamma[1:]
        return loss + pen, np.concatenate([gB.ravel(), ggamma])

    def fit(self, Qs: list[np.ndarray] | np.ndarray, y: np.ndarray, Z: np.ndarray | None = None):
        """Fit the stack on out-of-fold base predictions.

        Parameters
        ----------
        Qs:
            List of ``K`` arrays ``(n, J)`` (or one array ``(n, K, J)``) of base quantiles.
        y:
            Realised values ``(n,)``.
        Z:
            Gate features ``(n, p)`` (NaN allowed; imputed by 0 after standardisation).

        Returns
        -------
        GatedQuantileStack
            ``self`` (fitted).

        """
        Qs = np.stack(Qs, axis=1) if isinstance(Qs, list) else np.asarray(Qs)
        y = np.asarray(y, dtype=float)
        n, self.K, _ = Qs.shape
        if self.gate and Z is not None:
            self._zmean = np.nanmean(Z, axis=0)
            self._zstd = np.nanstd(Z, axis=0) + 1e-9
        Phi = self._design(Z, n)
        self.d = Phi.shape[1]
        eq = Qs.mean(axis=1)
        diff = eq - y[:, None]
        self._norm = float(np.where(diff > 0, (1 - self.levels) * diff, -self.levels * diff).mean())
        theta0 = np.zeros(self.K * self.d + self.d)
        res = minimize(
            self._loss_grad,
            theta0,
            args=(Phi, Qs, y),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.theta_, self.opt_ = res.x, res
        return self

    def weights(self, Qs: list[np.ndarray] | np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """Return the per-row base-learner weights ``(n, K)``."""
        Qs = np.stack(Qs, axis=1) if isinstance(Qs, list) else np.asarray(Qs)
        Phi = self._design(Z, len(Qs))
        return self._forward(self.theta_, Phi, Qs)[0]

    def predict(self, Qs: list[np.ndarray] | np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """Combine base quantiles into the stacked quantile forecast ``(n, J)``."""
        Qs = np.stack(Qs, axis=1) if isinstance(Qs, list) else np.asarray(Qs)
        Phi = self._design(Z, len(Qs))
        return sort_quantiles(self._forward(self.theta_, Phi, Qs)[3])


def equal_weight(Qs: list[np.ndarray]) -> np.ndarray:
    """Equal-weight quantile average (the baseline combination)."""
    return np.mean(np.stack(Qs, axis=0), axis=0)


def gate_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract the gate-feature matrix (float, NaN preserved) from a dataset frame."""
    return df[cols].astype(float).to_numpy()


#: Default regime descriptors fed to the gate (methodology Section 5.2.3).
GATE_COLS = [
    "lead_h",
    "ind_missing",
    "ind_imb_age_h",
    "abs_ind_imb",
    "niv_s48",
    "niv_s336",
    "wind_fc",
    "resload",
    "margin_surplus",
    "pod_sin",
    "pod_cos",
    "is_wkend",
]
