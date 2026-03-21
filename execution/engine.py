"""
Execution Engine with Transaction Cost Modeling.

Simulates realistic trade execution including:
    - Proportional transaction costs (commissions + spread)
    - Market impact (square-root model)
    - Slippage from execution delay
    - Turnover tracking

Mathematical Intuition:
    Total cost = commission + spread + market_impact
    Market impact ≈ σ * sqrt(Q / ADV) * sign(Q)
    where Q = trade quantity, ADV = average daily volume.

    The square-root model (Almgren & Chriss) is the industry
    standard for estimating market impact. It captures the
    empirical observation that impact scales sub-linearly
    with trade size.

Hedge Fund Usage:
    - Transaction costs are the #1 killer of alpha.
    - A strategy with 5bps/day alpha but 10bps round-trip costs
      is actually losing money.
    - Smart execution (VWAP, TWAP, implementation shortfall)
      can save 30-50% of market impact costs.
"""
import logging

import numpy as np
import pandas as pd

from config import ExecutionConfig

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Simulates trade execution with realistic cost modeling.
    Tracks portfolio positions, P&L, and turnover.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.positions: pd.Series | None = None
        self.cash = 0.0
        self.trade_log: list[dict] = []
        self.total_costs = 0.0
        self.total_turnover = 0.0

    def initialize(self, capital: float, assets: list[str]) -> None:
        """Set up initial portfolio state."""
        self.cash = capital
        self.positions = pd.Series(0.0, index=assets)

    def execute_rebalance(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        volumes: pd.Series | None = None,
        date: pd.Timestamp | None = None,
    ) -> dict:
        """
        Execute trades to move from current positions to target weights.

        Args:
            target_weights: Target portfolio weights.
            prices: Current asset prices.
            volumes: Current trading volumes (for impact estimation).
            date: Current date (for logging).

        Returns:
            Dict with trade details, costs, and new positions.
        """
        if self.positions is None:
            raise RuntimeError("ExecutionEngine not initialized")

        # Current portfolio value
        port_value = self.cash + (self.positions * prices).sum()
        if port_value <= 0:
            logger.warning("Portfolio value <= 0, skipping rebalance")
            return {"trades": {}, "costs": 0.0, "port_value": port_value}

        # Target positions in dollar terms
        target_dollars = target_weights * port_value
        # Target shares (continuous for simplicity)
        target_positions = target_dollars / prices.replace(0, np.nan)
        target_positions = target_positions.fillna(0)

        # Align with current positions
        common = self.positions.index.intersection(target_positions.index)
        current = self.positions[common]
        target = target_positions[common]

        # Trades needed
        trades = target - current
        trade_dollars = trades * prices[common]

        # Filter small trades (not worth the cost)
        min_trade = self.config.min_trade_size * port_value
        small_mask = trade_dollars.abs() < min_trade
        trades[small_mask] = 0.0
        trade_dollars[small_mask] = 0.0

        # Compute costs for each trade
        total_cost = 0.0
        trade_details = {}

        for asset in common:
            if abs(trades[asset]) < 1e-10:
                continue

            trade_value = abs(trade_dollars[asset])

            # Commission cost
            commission = trade_value * self.config.commission_rate

            # Spread cost (half-spread per side)
            spread_cost = trade_value * self.config.spread_cost

            # Market impact (square-root model)
            impact = 0.0
            if volumes is not None and asset in volumes.index:
                adv = volumes[asset] * prices[asset]  # Dollar volume
                if adv > 0:
                    participation = trade_value / adv
                    daily_vol = 0.02  # Assume ~2% daily vol as default
                    impact = daily_vol * np.sqrt(participation) * trade_value

            cost = commission + spread_cost + impact

            trade_details[asset] = {
                "shares": trades[asset],
                "dollars": trade_dollars[asset],
                "commission": commission,
                "spread": spread_cost,
                "impact": impact,
                "total_cost": cost,
            }

            total_cost += cost

        # Apply trades
        for asset, detail in trade_details.items():
            self.positions[asset] += detail["shares"]
            self.cash -= detail["dollars"] + detail["total_cost"]

        self.total_costs += total_cost
        turnover = sum(abs(d["dollars"]) for d in trade_details.values()) / port_value
        self.total_turnover += turnover

        # Log
        if date is not None:
            self.trade_log.append({
                "date": date,
                "n_trades": len(trade_details),
                "turnover": turnover,
                "total_cost": total_cost,
                "port_value": port_value,
            })

        return {
            "trades": trade_details,
            "costs": total_cost,
            "turnover": turnover,
            "port_value": port_value + (self.positions * prices).sum() - port_value,
        }

    def get_portfolio_value(self, prices: pd.Series) -> float:
        """Compute current portfolio value."""
        if self.positions is None:
            return self.cash
        return self.cash + (self.positions * prices).sum()

    def get_weights(self, prices: pd.Series) -> pd.Series:
        """Compute current portfolio weights."""
        port_value = self.get_portfolio_value(prices)
        if port_value <= 0 or self.positions is None:
            return pd.Series(dtype=float)
        return (self.positions * prices) / port_value

    def get_execution_summary(self) -> dict:
        """Return summary of execution statistics."""
        return {
            "total_costs": self.total_costs,
            "total_turnover": self.total_turnover,
            "n_rebalances": len(self.trade_log),
            "avg_turnover": (
                self.total_turnover / len(self.trade_log)
                if self.trade_log else 0.0
            ),
            "avg_cost_per_rebalance": (
                self.total_costs / len(self.trade_log)
                if self.trade_log else 0.0
            ),
        }
