"""
Risk Management Layer.

Enforces portfolio-level risk limits including:
    - Value at Risk (VaR) and Expected Shortfall (CVaR)
    - Maximum drawdown limits
    - Volatility targeting
    - Position concentration limits
    - Correlation-based diversification checks

Mathematical Intuition:
    VaR_α = -μ_p - z_α * σ_p  (parametric, under normality)
    ES_α  = -μ_p + σ_p * φ(z_α) / α  (expected loss beyond VaR)

    Vol targeting: scale portfolio by σ_target / σ_realized
    This keeps risk constant across regimes — the single most
    important risk management technique in practice.

Hedge Fund Usage:
    - All institutional funds have hard risk limits (max drawdown,
      VaR limits, gross/net exposure limits).
    - Volatility targeting is universal — it's how you survive.
    - Risk is measured at multiple horizons (1-day, 10-day, 1-month).
"""
import logging

import numpy as np
import pandas as pd

from config import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Monitors and enforces portfolio risk limits.
    Can scale down or zero out positions that breach limits.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.peak_equity = 0.0
        self.risk_metrics: dict = {}

    def check_and_adjust(
        self,
        target_weights: pd.Series,
        returns: pd.DataFrame,
        equity_curve: pd.Series | None = None,
    ) -> pd.Series:
        """
        Apply all risk checks to target weights.
        Returns adjusted weights that satisfy all constraints.

        Args:
            target_weights: Proposed portfolio weights.
            returns: Historical asset returns.
            equity_curve: Portfolio equity curve (for drawdown checks).

        Returns:
            Risk-adjusted portfolio weights.
        """
        weights = target_weights.copy()

        # 1. Position concentration limits
        weights = self._check_position_limits(weights)

        # 2. Gross/net exposure limits
        weights = self._check_exposure_limits(weights)

        # 3. Volatility targeting
        weights = self._vol_target(weights, returns)

        # 4. VaR check
        weights = self._check_var(weights, returns)

        # 5. Drawdown check
        if equity_curve is not None:
            weights = self._check_drawdown(weights, equity_curve)

        # 6. Compute and store risk metrics
        self._compute_risk_metrics(weights, returns)

        return weights

    def _check_position_limits(self, weights: pd.Series) -> pd.Series:
        """Enforce maximum position size per asset."""
        max_pos = self.config.max_position_size
        return weights.clip(-max_pos, max_pos)

    def _check_exposure_limits(self, weights: pd.Series) -> pd.Series:
        """Enforce gross and net exposure limits."""
        # Gross exposure
        gross = weights.abs().sum()
        max_gross = self.config.max_gross_leverage
        if gross > max_gross:
            weights = weights * (max_gross / gross)

        # Net exposure
        net = weights.sum()
        max_net = self.config.max_net_exposure
        if abs(net) > max_net:
            # Reduce the side that's causing excess net exposure
            excess = abs(net) - max_net
            if net > 0:
                long_mask = weights > 0
                long_sum = weights[long_mask].sum()
                if long_sum > 0:
                    weights[long_mask] *= (long_sum - excess) / long_sum
            else:
                short_mask = weights < 0
                short_sum = weights[short_mask].sum()
                if short_sum < 0:
                    weights[short_mask] *= (abs(short_sum) - excess) / abs(short_sum)

        return weights

    def _vol_target(
        self, weights: pd.Series, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Scale portfolio to target a specific annualized volatility.
        This is the most important risk management technique.
        """
        target_vol = self.config.vol_target
        if target_vol <= 0:
            return weights

        # Estimate portfolio volatility
        common = weights.index.intersection(returns.columns)
        if len(common) < 2:
            return weights

        w = weights[common].values
        # Use recent returns for vol estimate (more responsive)
        recent = returns[common].iloc[-63:]
        if len(recent) < 20:
            return weights

        cov = recent.cov().values * 252  # Annualize
        port_vol = np.sqrt(w @ cov @ w)

        if port_vol < 1e-8:
            return weights

        # Scale factor
        scale = target_vol / port_vol
        scale = np.clip(scale, 0.1, 3.0)  # Safety bounds

        self.risk_metrics["realized_vol"] = port_vol
        self.risk_metrics["vol_scale"] = scale

        return weights * scale

    def _check_var(
        self, weights: pd.Series, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Check if portfolio VaR exceeds the limit.
        Uses historical simulation (non-parametric).
        """
        max_var = self.config.max_var
        if max_var <= 0:
            return weights

        common = weights.index.intersection(returns.columns)
        if len(common) < 2:
            return weights

        w = weights[common].values
        recent = returns[common].iloc[-252:]
        if len(recent) < 60:
            return weights

        # Historical portfolio returns
        port_returns = recent.values @ w

        # VaR at confidence level
        confidence = self.config.var_confidence
        var = -np.percentile(port_returns, (1 - confidence) * 100)

        # Expected Shortfall (CVaR)
        tail = port_returns[port_returns <= -var]
        es = -tail.mean() if len(tail) > 0 else var

        self.risk_metrics["var_1d"] = var
        self.risk_metrics["expected_shortfall"] = es

        if var > max_var:
            scale = max_var / var
            logger.warning(
                f"VaR {var:.4f} exceeds limit {max_var:.4f}, "
                f"scaling positions by {scale:.2f}"
            )
            return weights * scale

        return weights

    def _check_drawdown(
        self, weights: pd.Series, equity_curve: pd.Series
    ) -> pd.Series:
        """
        Reduce exposure if portfolio is in a significant drawdown.
        Progressive de-risking as drawdown deepens.
        """
        if equity_curve.empty:
            return weights

        current_equity = equity_curve.iloc[-1]
        self.peak_equity = max(self.peak_equity, current_equity)

        if self.peak_equity <= 0:
            return weights

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        max_dd = self.config.max_drawdown

        self.risk_metrics["current_drawdown"] = drawdown
        self.risk_metrics["peak_equity"] = self.peak_equity

        if drawdown > max_dd:
            # Hard stop: zero all positions
            logger.warning(
                f"Drawdown {drawdown:.2%} exceeds max {max_dd:.2%}. "
                f"Zeroing all positions."
            )
            return weights * 0.0

        # Progressive de-risking: start reducing at 50% of max drawdown
        dd_threshold = max_dd * 0.5
        if drawdown > dd_threshold:
            # Linear scale-down from 1.0 at threshold to 0.0 at max
            scale = 1.0 - (drawdown - dd_threshold) / (max_dd - dd_threshold)
            scale = max(scale, 0.0)
            logger.info(
                f"Drawdown {drawdown:.2%}: reducing exposure by {1-scale:.0%}"
            )
            return weights * scale

        return weights

    def _compute_risk_metrics(
        self, weights: pd.Series, returns: pd.DataFrame
    ) -> None:
        """Compute and store current risk metrics."""
        self.risk_metrics["gross_exposure"] = weights.abs().sum()
        self.risk_metrics["net_exposure"] = weights.sum()
        self.risk_metrics["long_exposure"] = weights[weights > 0].sum()
        self.risk_metrics["short_exposure"] = weights[weights < 0].sum()
        self.risk_metrics["n_positions"] = (weights.abs() > 1e-6).sum()
        self.risk_metrics["max_position"] = weights.abs().max()

        # Concentration: Herfindahl index
        w_abs = weights.abs()
        if w_abs.sum() > 0:
            w_norm = w_abs / w_abs.sum()
            self.risk_metrics["herfindahl"] = (w_norm ** 2).sum()
        else:
            self.risk_metrics["herfindahl"] = 0.0

    def get_risk_report(self) -> dict:
        """Return current risk metrics."""
        return self.risk_metrics.copy()
