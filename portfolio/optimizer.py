"""
Portfolio Construction / Optimization Layer.

Converts alpha signals into target portfolio weights using
mean-variance optimization, risk parity, or Black-Litterman.

Mathematical Intuition:
    Mean-Variance (Markowitz):
        max  w'μ - (λ/2) w'Σw
        s.t. Σw_i = 1, |w_i| ≤ max_weight

    Risk Parity:
        Each asset contributes equally to total portfolio risk.
        w_i ∝ 1 / σ_i (simplified), solved iteratively for exact.

    Black-Litterman:
        Combines market equilibrium returns (from CAPM) with
        alpha views to produce a blended expected return vector.
        Π = λΣw_mkt (equilibrium returns)
        E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 Π + P'Ω^-1 Q]

Hedge Fund Usage:
    - Most funds use constrained mean-variance or risk parity.
    - Black-Litterman is popular because it produces stable,
      diversified portfolios even with noisy alpha signals.
    - Position limits and sector constraints are always applied.
"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import PortfolioConfig

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Constructs target portfolio weights from alpha signals
    and a covariance estimate.
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def optimize(
        self,
        signals: pd.Series,
        cov_matrix: pd.DataFrame,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Compute target portfolio weights.

        Args:
            signals: Expected return signals per asset (cross-sectional).
            cov_matrix: Covariance matrix of asset returns.
            current_weights: Current portfolio weights (for turnover penalty).

        Returns:
            Target weight per asset summing to <= 1 (gross).
        """
        assets = signals.index
        n = len(assets)

        # Align covariance matrix with signal assets
        common = cov_matrix.index.intersection(assets)
        if len(common) < 2:
            return pd.Series(0.0, index=assets)

        signals = signals[common]
        cov = cov_matrix.loc[common, common]

        if self.config.method == "mean_variance":
            weights = self._mean_variance(signals, cov, current_weights)
        elif self.config.method == "risk_parity":
            weights = self._risk_parity(cov)
            # Tilt risk parity weights by signal direction
            weights = weights * np.sign(signals)
        elif self.config.method == "black_litterman":
            weights = self._black_litterman(signals, cov)
        else:
            weights = self._mean_variance(signals, cov, current_weights)

        # Apply position limits
        weights = self._apply_constraints(weights)

        # Reindex to full asset universe
        full_weights = pd.Series(0.0, index=assets)
        full_weights[weights.index] = weights
        return full_weights

    def _mean_variance(
        self,
        signals: pd.Series,
        cov: pd.DataFrame,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Constrained mean-variance optimization.
        max  w'μ - (λ/2) w'Σw - tc * |w - w_prev|
        """
        n = len(signals)
        mu = signals.values
        sigma = cov.values
        risk_aversion = self.config.risk_aversion

        # Regularize covariance (Ledoit-Wolf shrinkage approximation)
        sigma = self._shrink_covariance(sigma)

        w0 = np.zeros(n)
        if current_weights is not None:
            common = signals.index.intersection(current_weights.index)
            for i, asset in enumerate(signals.index):
                if asset in current_weights.index:
                    w0[i] = current_weights[asset]

        tc_penalty = self.config.turnover_penalty

        def objective(w):
            ret = w @ mu
            risk = w @ sigma @ w
            turnover = np.sum(np.abs(w - w0))
            return -(ret - (risk_aversion / 2) * risk - tc_penalty * turnover)

        max_w = self.config.max_position_size
        bounds = [(-max_w, max_w)] * n

        # Gross leverage constraint
        max_leverage = self.config.max_gross_leverage
        constraints = [
            {"type": "ineq", "fun": lambda w: max_leverage - np.sum(np.abs(w))},
        ]

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if result.success:
            return pd.Series(result.x, index=signals.index)
        else:
            logger.warning(f"Optimization failed: {result.message}. Using signal-proportional weights.")
            return self._signal_proportional(signals)

    def _risk_parity(self, cov: pd.DataFrame) -> pd.Series:
        """
        Risk parity: equal risk contribution from each asset.
        Solved via optimization: min Σ_i (w_i (Σw)_i - σ_p²/n)²
        """
        n = len(cov)
        sigma = self._shrink_covariance(cov.values)

        def risk_contribution(w):
            port_var = w @ sigma @ w
            marginal = sigma @ w
            rc = w * marginal
            return rc

        def objective(w):
            rc = risk_contribution(w)
            target_rc = np.sum(rc) / n
            return np.sum((rc - target_rc) ** 2)

        w0 = np.ones(n) / n
        bounds = [(0.001, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

        if result.success:
            return pd.Series(result.x, index=cov.index)
        else:
            # Fallback: inverse volatility
            vols = np.sqrt(np.diag(sigma))
            inv_vol = 1.0 / (vols + 1e-8)
            weights = inv_vol / inv_vol.sum()
            return pd.Series(weights, index=cov.index)

    def _black_litterman(
        self, signals: pd.Series, cov: pd.DataFrame
    ) -> pd.Series:
        """
        Black-Litterman model.

        Combines market equilibrium returns with alpha views.
        Uses signals as absolute views (P = I, Q = signals).
        """
        n = len(signals)
        sigma = self._shrink_covariance(cov.values)

        # Market equilibrium: assume equal-weight market portfolio
        w_mkt = np.ones(n) / n
        risk_aversion = self.config.risk_aversion
        tau = 0.05  # Uncertainty in equilibrium returns

        # Equilibrium expected returns
        pi = risk_aversion * sigma @ w_mkt

        # Views: P = identity (absolute views), Q = signals
        P = np.eye(n)
        Q = signals.values

        # View uncertainty: proportional to asset variance
        omega = np.diag(np.diag(tau * sigma))

        # Posterior expected returns
        tau_sigma_inv = np.linalg.inv(tau * sigma)
        omega_inv = np.linalg.inv(omega)

        posterior_cov = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        posterior_mu = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        # Now optimize with posterior returns
        posterior_signals = pd.Series(posterior_mu, index=signals.index)
        posterior_cov_df = pd.DataFrame(sigma + posterior_cov, index=cov.index, columns=cov.columns)

        return self._mean_variance(posterior_signals, posterior_cov_df)

    def _signal_proportional(self, signals: pd.Series) -> pd.Series:
        """Fallback: weights proportional to signal magnitude."""
        abs_sum = signals.abs().sum()
        if abs_sum < 1e-8:
            return pd.Series(0.0, index=signals.index)
        weights = signals / abs_sum * self.config.max_gross_leverage
        return weights.clip(-self.config.max_position_size, self.config.max_position_size)

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply position size and leverage constraints."""
        max_pos = self.config.max_position_size
        weights = weights.clip(-max_pos, max_pos)

        # Scale down if gross leverage exceeds limit
        gross = weights.abs().sum()
        if gross > self.config.max_gross_leverage:
            weights = weights * (self.config.max_gross_leverage / gross)

        return weights

    @staticmethod
    def _shrink_covariance(sigma: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
        """
        Ledoit-Wolf-style shrinkage toward diagonal.
        Improves conditioning of the covariance matrix.
        """
        n = sigma.shape[0]
        target = np.diag(np.diag(sigma))
        return (1 - shrinkage) * sigma + shrinkage * target

    @staticmethod
    def estimate_covariance(
        returns: pd.DataFrame, method: str = "exponential", halflife: int = 63
    ) -> pd.DataFrame:
        """
        Estimate the covariance matrix from returns.

        Args:
            returns: Asset returns DataFrame.
            method: 'sample', 'exponential', or 'shrinkage'.
            halflife: For exponential weighting.
        """
        if method == "exponential":
            # Exponentially weighted covariance (more responsive to recent data)
            ewm = returns.ewm(halflife=halflife)
            cov = ewm.cov().iloc[-len(returns.columns):]
            # ewm.cov() returns a multi-indexed DataFrame; extract the last block
            last_date = returns.index[-1]
            try:
                cov = returns.ewm(halflife=halflife).cov()
                cov = cov.loc[last_date]
            except (KeyError, TypeError):
                cov = returns.cov()
        elif method == "shrinkage":
            sample_cov = returns.cov().values
            n = sample_cov.shape[0]
            target = np.diag(np.diag(sample_cov))
            shrunk = 0.85 * sample_cov + 0.15 * target
            cov = pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
        else:
            cov = returns.cov()

        return cov
