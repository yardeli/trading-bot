"""
Configuration for the quantitative trading engine.
All tunable parameters live here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Universe of tradeable assets
    tickers: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",   # Equity indices
        "TLT", "IEF", "HYG", "LQD",           # Fixed income
        "GLD", "SLV", "USO", "DBA",            # Commodities
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",  # Mega-cap tech
        "JPM", "GS", "BAC",                    # Financials
        "XOM", "CVX",                           # Energy
    ])
    years: int = 5  # Years of historical data
    min_history_days: int = 252  # Require at least 1 year
    price_field: str = "Adj Close"


@dataclass
class FeatureConfig:
    # Momentum lookback windows
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63, 126, 252])
    # Volatility estimation windows
    vol_windows: List[int] = field(default_factory=lambda: [10, 21, 63])
    # Mean reversion lookback
    mean_rev_windows: List[int] = field(default_factory=lambda: [5, 10, 21])
    # RSI period
    rsi_period: int = 14
    # Bollinger Band window
    bb_window: int = 20
    bb_std: float = 2.0


@dataclass
class AlphaConfig:
    # Time-series momentum
    momentum_fast: int = 10
    momentum_slow: int = 63
    momentum_vol_target: float = 0.10  # 10% annualized vol target

    # Mean reversion
    mean_rev_halflife: int = 21
    mean_rev_entry_z: float = 1.0
    mean_rev_exit_z: float = 0.5

    # Statistical arbitrage
    stat_arb_formation_period: int = 252
    stat_arb_trading_period: int = 63
    stat_arb_entry_z: float = 1.5
    stat_arb_exit_z: float = 0.5
    stat_arb_stop_z: float = 4.0

    # ML Alpha
    ml_train_window: int = 504  # 2 years
    ml_retrain_frequency: int = 42  # ~bimonthly, more adaptive
    ml_n_estimators: int = 200
    ml_max_depth: int = 4
    ml_min_samples_leaf: int = 20
    ml_forward_return_horizon: int = 10  # 2-week forward returns (matches rebalance freq)

    # Ensemble
    ensemble_method: str = "performance_weighted"  # IC-based: reward models that predict returns
    enabled_models: List[str] = field(default_factory=lambda: [
        "ts_momentum", "xs_momentum", "momentum_vol_break",
        "ou_mean_reversion", "pairs_trading", "ml_alpha",
    ])


@dataclass
class PortfolioConfig:
    # Optimization method: 'mean_variance', 'risk_parity', 'black_litterman', 'equal_weight'
    method: str = "mean_variance"
    # Constraints
    max_position_size: float = 0.20  # 15% max in any single name
    max_sector_weight: float = 0.40
    max_gross_leverage: float = 1.5
    max_net_leverage: float = 0.6  # Allow more directional exposure
    turnover_penalty: float = 0.005  # 50bps turnover cost
    risk_aversion: float = 1.5  # Lambda for mean-variance optimization
    # Rebalance frequency (trading days)
    rebalance_frequency: int = 10


@dataclass
class RiskConfig:
    # Drawdown limits
    max_drawdown: float = 0.25  # 25% max drawdown (de-risking starts at 12.5%)

    # VaR limits
    var_confidence: float = 0.99
    max_var: float = 0.035  # 3.5% daily VaR limit

    # Volatility targeting
    vol_target: float = 0.15  # 15% annualized

    # Position / exposure limits
    max_position_size: float = 0.20  # 15% max single name
    max_gross_leverage: float = 1.5
    max_net_exposure: float = 0.6  # Allow more directional exposure


@dataclass
class ExecutionConfig:
    # Transaction costs (one-way, as decimals)
    commission_rate: float = 0.0001  # 1 bps
    spread_cost: float = 0.0002  # 2 bps half-spread
    # Minimum trade size as fraction of portfolio
    min_trade_size: float = 0.001  # 0.1% of portfolio


@dataclass
class BacktestConfig:
    # Walk-forward settings
    warmup_period: int = 252  # 1 year warmup before trading
    rebalance_frequency: int = 10  # Rebalance every 5 trading days
    # Starting capital
    initial_capital: float = 1_000_000.0


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
