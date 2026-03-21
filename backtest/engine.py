"""
Walk-Forward Backtesting Engine.

Simulates the full trading pipeline on historical data:
    Data → Features → Alpha → Ensemble → Portfolio → Risk → Execution

Key Design Principles:
    - Walk-forward only: no future information leaks.
    - Features, signals, and portfolio weights are computed
      using only data available at each point in time.
    - Transaction costs are modeled realistically.
    - Performance is measured with proper risk-adjusted metrics.

Hedge Fund Usage:
    - Walk-forward backtesting is the minimum standard.
    - In-sample / out-of-sample splits prevent overfitting.
    - Monte Carlo permutation tests verify statistical significance.
    - Realistic transaction costs separate paper alpha from real alpha.
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from alpha.base import AlphaModel
from config import SystemConfig
from data.feed import DataFeed
from ensemble.aggregator import SignalAggregator
from execution.engine import ExecutionEngine
from features.engine import FeatureEngine
from portfolio.optimizer import PortfolioOptimizer
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    trade_log: list[dict]
    risk_metrics_history: list[dict]
    execution_summary: dict
    performance_metrics: dict


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Runs the complete pipeline day by day on historical data,
    respecting the temporal ordering to prevent look-ahead bias.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.feature_engine = FeatureEngine(config.features)
        self.signal_aggregator = SignalAggregator(
            method=config.alpha.ensemble_method,
            ic_lookback=63,
        )
        self.portfolio_optimizer = PortfolioOptimizer(config.portfolio)
        self.risk_manager = RiskManager(config.risk)
        self.execution_engine = ExecutionEngine(config.execution)

    def run(
        self,
        data: DataFeed,
        alpha_models: list[AlphaModel],
        initial_capital: float = 1_000_000,
        dashboard=None,
    ) -> BacktestResult:
        """
        Run a full walk-forward backtest.

        Args:
            data: Historical market data.
            alpha_models: List of alpha models to run.
            initial_capital: Starting capital.
            dashboard: Optional TerminalDashboard for live UI updates.

        Returns:
            BacktestResult with equity curve, metrics, etc.
        """
        prices = data.prices
        returns = data.returns
        dates = prices.index
        assets = prices.columns.tolist()

        # Initialize execution engine
        self.execution_engine.initialize(initial_capital, assets)

        # Compute features once (they only use past data internally)
        logger.info("Computing features...")
        features = self.feature_engine.generate(data)

        # Generate signals from each alpha model
        logger.info("Generating alpha signals...")
        all_signals = {}
        for model in alpha_models:
            logger.info(f"  Running {model.name}...")
            sig = model.generate_signals(data, features)
            all_signals[model.name] = sig

        # Determine rebalance dates
        warmup = self.config.backtest.warmup_period
        rebal_freq = self.config.backtest.rebalance_frequency
        rebalance_dates = dates[warmup::rebal_freq]

        logger.info(
            f"Backtesting {len(rebalance_dates)} rebalance dates "
            f"({dates[warmup]} to {dates[-1]})"
        )

        # Walk-forward simulation
        equity_values = []
        weights_history = []
        risk_metrics_history = []

        current_weights = pd.Series(0.0, index=assets)

        # Pre-convert rebalance dates to a set for O(1) lookup
        rebalance_set = set(rebalance_dates.values)

        # Build equity curve incrementally (avoid rebuilding dict each rebalance)
        equity_dates = []
        equity_vals = []

        # Max lookback for signals/returns passed to aggregator (prevents growing slices)
        AGG_LOOKBACK = 300  # ~1.2 years of trading days

        for i, date in enumerate(dates):
            current_prices = prices.loc[date]

            # Record equity
            port_value = self.execution_engine.get_portfolio_value(current_prices)
            equity_values.append({"date": date, "equity": port_value})
            equity_dates.append(date)
            equity_vals.append(port_value)

            is_rebalance = date in rebalance_set
            trade_info = None

            # Rebalance if scheduled
            if is_rebalance:
                # Get signals for this date — only recent lookback, not full history
                date_signals = {}
                for name, sig_df in all_signals.items():
                    if date in sig_df.index:
                        date_idx = sig_df.index.get_loc(date)
                        start_idx = max(0, date_idx - AGG_LOOKBACK + 1)
                        date_signals[name] = sig_df.iloc[start_idx:date_idx + 1]

                if date_signals:
                    # Aggregate signals — pass only recent returns
                    try:
                        ret_end = returns.index.get_loc(date)
                        ret_start = max(0, ret_end - AGG_LOOKBACK + 1)
                        recent_returns = returns.iloc[ret_start:ret_end + 1]

                        combined = self.signal_aggregator.aggregate(
                            date_signals, recent_returns
                        )
                        signal_row = combined.iloc[-1]
                    except Exception as e:
                        logger.debug(f"Signal aggregation failed at {date}: {e}")
                        signal_row = None

                    if signal_row is not None:
                        # Estimate covariance from recent returns (already bounded)
                        lookback_returns = returns.loc[:date].iloc[-252:]
                        if len(lookback_returns) >= 60:
                            cov_matrix = PortfolioOptimizer.estimate_covariance(
                                lookback_returns,
                                method="exponential",
                                halflife=63,
                            )

                            # Optimize portfolio
                            try:
                                target_weights = self.portfolio_optimizer.optimize(
                                    signal_row, cov_matrix, current_weights
                                )
                            except Exception as e:
                                logger.debug(f"Optimization failed at {date}: {e}")
                                target_weights = None

                            if target_weights is not None:
                                # Risk checks — use incrementally built equity curve
                                equity_curve_so_far = pd.Series(
                                    equity_vals, index=equity_dates
                                )
                                target_weights = self.risk_manager.check_and_adjust(
                                    target_weights, lookback_returns, equity_curve_so_far
                                )

                                # Execute trades
                                volumes = data.volume.loc[date] if data.volume is not None else None
                                self.execution_engine.execute_rebalance(
                                    target_weights, current_prices, volumes, date
                                )

                                current_weights = self.execution_engine.get_weights(current_prices)

                                # Record
                                weights_history.append({"date": date, **current_weights.to_dict()})
                                risk_metrics_history.append(
                                    {"date": date, **self.risk_manager.get_risk_report()}
                                )

                                if self.execution_engine.trade_log:
                                    trade_info = self.execution_engine.trade_log[-1]

            # Dashboard update (once per day, after any rebalance)
            if dashboard is not None:
                day_kwargs = {"date": date, "equity": port_value}
                if is_rebalance and current_weights is not None:
                    day_kwargs["weights"] = {
                        k: v for k, v in current_weights.items() if abs(v) > 0.001
                    }
                if self.risk_manager.risk_metrics:
                    day_kwargs["risk_metrics"] = self.risk_manager.get_risk_report()
                if self.signal_aggregator.model_weights:
                    day_kwargs["model_weights"] = self.signal_aggregator.model_weights
                if trade_info is not None:
                    day_kwargs["trade_info"] = trade_info
                dashboard.on_day_update(**day_kwargs)

        # Notify dashboard
        if dashboard is not None:
            dashboard.on_backtest_complete()

        # Build results
        equity_df = pd.DataFrame(equity_values).set_index("date")["equity"]
        port_returns = equity_df.pct_change().dropna()

        weights_df = pd.DataFrame(weights_history)
        if not weights_df.empty:
            weights_df = weights_df.set_index("date")

        performance = self._compute_performance_metrics(
            equity_df, port_returns, initial_capital
        )

        result = BacktestResult(
            equity_curve=equity_df,
            returns=port_returns,
            weights_history=weights_df,
            trade_log=self.execution_engine.trade_log,
            risk_metrics_history=risk_metrics_history,
            execution_summary=self.execution_engine.get_execution_summary(),
            performance_metrics=performance,
        )

        # Show final report in dashboard or print to console
        if dashboard is not None:
            dashboard.show_final_report(result)
        else:
            self._print_summary(result)

        return result

    def _compute_performance_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        initial_capital: float,
    ) -> dict:
        """Compute comprehensive performance statistics."""
        if returns.empty or len(returns) < 2:
            return {}

        total_return = (equity.iloc[-1] / initial_capital) - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Sortino ratio (downside deviation only)
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Win rate
        win_rate = (returns > 0).mean()

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float("inf")

        # Skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "skewness": skew,
            "kurtosis": kurt,
            "n_trading_days": len(returns),
            "final_equity": equity.iloc[-1],
        }

    @staticmethod
    def _print_summary(result: BacktestResult) -> None:
        """Print a formatted performance summary."""
        m = result.performance_metrics
        if not m:
            logger.warning("No performance metrics to display")
            return

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total Return:        {m['total_return']:>10.2%}")
        print(f"  Annualized Return:   {m['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {m['annualized_volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {m['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {m['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {m['max_drawdown']:>10.2%}")
        print(f"  Calmar Ratio:        {m['calmar_ratio']:>10.2f}")
        print(f"  Win Rate:            {m['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {m['profit_factor']:>10.2f}")
        print(f"  Skewness:            {m['skewness']:>10.2f}")
        print(f"  Kurtosis:            {m['kurtosis']:>10.2f}")
        print(f"  Trading Days:        {m['n_trading_days']:>10d}")
        print(f"  Final Equity:        ${m['final_equity']:>12,.2f}")
        print("-" * 60)

        ex = result.execution_summary
        print(f"  Total Costs:         ${ex['total_costs']:>12,.2f}")
        print(f"  Total Turnover:      {ex['total_turnover']:>10.2f}x")
        print(f"  Rebalances:          {ex['n_rebalances']:>10d}")
        print("=" * 60 + "\n")

    def export_brain(
        self,
        result: "BacktestResult",
        output_path: str | None = None,
    ) -> dict:
        """
        Export backtest knowledge as a brain_export.json compatible with
        paper-trader-v4.html's adaptive brain system.

        Maps quant-engine model performance → paper-trader strategy weights
        so the live trader starts pre-trained instead of blank.
        """
        perf = result.performance_metrics
        model_weights = self.signal_aggregator.model_weights or {}

        # ── Map QE models → paper-trader strategies ──────────────
        # QE models: ts_momentum, xs_momentum, momentum_vol_break,
        #            ou_mean_reversion, pairs_trading, ml_alpha
        # Paper-trader: RSI, MA, Momentum, MeanRev, VWAP, Composite,
        #               Stochastic, Bollinger
        #
        # Mapping logic:
        #   ts_momentum     → Momentum (time-series trend following)
        #   xs_momentum     → Composite (cross-sectional = multi-factor)
        #   momentum_vol_break → MA (trend + vol regime = MA crossover style)
        #   ou_mean_reversion  → MeanRev, RSI, Bollinger (mean reversion family)
        #   pairs_trading      → Stochastic (relative value / oversold)
        #   ml_alpha           → VWAP, Composite (nonlinear multi-factor)

        qe_to_pt = {
            "Momentum": ["ts_momentum"],
            "MA": ["momentum_vol_break"],
            "Composite": ["xs_momentum", "ml_alpha"],
            "MeanRev": ["ou_mean_reversion"],
            "RSI": ["ou_mean_reversion"],
            "Bollinger": ["ou_mean_reversion"],
            "Stochastic": ["pairs_trading"],
            "VWAP": ["ml_alpha"],
        }

        strategy_weights = {}
        for pt_strat, qe_models in qe_to_pt.items():
            # Average the QE model weights that map to this strategy
            weights = [model_weights.get(m, 0) for m in qe_models]
            avg_w = sum(weights) / len(weights) if weights else 0
            # Floor at 0.15 so no strategy is completely dead
            strategy_weights[pt_strat] = max(avg_w, 0.15)

        # Normalize so average weight = 1.0
        if strategy_weights:
            mean_w = sum(strategy_weights.values()) / len(strategy_weights)
            if mean_w > 0:
                strategy_weights = {
                    k: round(max(0.3, v / mean_w), 4)
                    for k, v in strategy_weights.items()
                }

        # ── Derive regime from backtest performance ──────────────
        # If Sharpe > 0.5, trending worked; if mean_rev models dominate, choppy
        mr_weight = model_weights.get("ou_mean_reversion", 0)
        mom_weight = sum(
            model_weights.get(m, 0)
            for m in ["ts_momentum", "xs_momentum", "momentum_vol_break"]
        )
        total_w = mr_weight + mom_weight
        if total_w > 0:
            trend_pct = mom_weight / total_w
        else:
            trend_pct = 0.5

        if trend_pct > 0.6:
            regime = "trending"
        elif trend_pct < 0.4:
            regime = "choppy"
        else:
            regime = "mixed"

        # ── Derive risk thresholds from backtest ─────────────────
        sharpe = perf.get("sharpe_ratio", 0)
        max_dd = abs(perf.get("max_drawdown", 0))
        win_rate = perf.get("win_rate", 0.5)

        # If backtest had strong performance, start more aggressive
        if sharpe > 0.8:
            sl_mult, tp_mult, entry_mult = 1.1, 1.15, 0.85
        elif sharpe > 0.4:
            sl_mult, tp_mult, entry_mult = 1.05, 1.05, 0.95
        elif sharpe < 0:
            sl_mult, tp_mult, entry_mult = 0.9, 0.9, 1.1
        else:
            sl_mult, tp_mult, entry_mult = 1.0, 1.0, 1.0

        # ── Build the brain export ───────────────────────────────
        strategies = {}
        for strat, weight in strategy_weights.items():
            strategies[strat] = {
                "weight": weight,
                "winRate": round(win_rate, 4),
                "avgPnl": 0,
                "avgPnlPct": round(perf.get("annualized_return", 0) / 252, 6),
                "trades": 0,  # Will be filled by live trading
                "recentResults": [],
            }

        # Get ML feature importance if available
        feature_importance = {}
        for model in getattr(self, '_alpha_models', []):
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                break

        brain_export = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "source": "quant-engine-backtest",
            "backtest_summary": {
                "total_return": round(perf.get("total_return", 0), 6),
                "annualized_return": round(perf.get("annualized_return", 0), 6),
                "sharpe_ratio": round(sharpe, 4),
                "sortino_ratio": round(perf.get("sortino_ratio", 0), 4),
                "max_drawdown": round(perf.get("max_drawdown", 0), 6),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(perf.get("profit_factor", 0), 4),
                "trading_days": perf.get("n_trading_days", 0),
            },
            "model_weights_raw": {
                k: round(v, 6) for k, v in model_weights.items()
            },
            "strategies": strategies,
            "thresholds": {
                "slMult": round(sl_mult, 4),
                "tpMult": round(tp_mult, 4),
                "entryMult": round(entry_mult, 4),
            },
            "regime": {
                "type": regime,
                "confidence": round(abs(trend_pct - 0.5) * 2, 4),
                "history": [],
            },
            "feature_importance": {
                k: round(v, 6) for k, v in list(feature_importance.items())[:20]
            },
            "streak": {"current": 0, "type": None},
            "learningEnabled": True,
            "totalLearningTrades": 0,
            "lastSaved": None,
        }

        # Save to file
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "brain_export.json"
            )
        with open(output_path, "w") as f:
            json.dump(brain_export, f, indent=2)

        logger.info(f"Brain exported to {output_path}")
        return brain_export
