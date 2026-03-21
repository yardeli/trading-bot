"""
Auto-Trader Configuration
=========================
Regime-Adaptive Crypto Multi-Strategy: Momentum + Mean Reversion
"""

# ── Assets ──────────────────────────────────────────────────────────
ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "MATIC-USD",
    "DOGE-USD", "ATOM-USD", "UNI-USD", "LTC-USD", "NEAR-USD",
]
BENCHMARK = "BTC-USD"

# ── Data ────────────────────────────────────────────────────────────
DATA_SOURCE = "yfinance"           # "yfinance" or "ccxt"
OHLCV_INTERVAL = "1d"             # daily bars
LOOKBACK_YEARS = 4                # years of history for training
MIN_HISTORY_DAYS = 365            # minimum days required per asset

# ── Feature Engineering ─────────────────────────────────────────────
RSI_PERIOD = 14
RSI_SHORT_PERIOD = 6              # short RSI for mean reversion
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
MOMENTUM_LOOKBACKS = [7, 14, 30, 60, 90]  # days
VOLATILITY_WINDOW = 20
ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20
EMA_PERIODS = [8, 21, 50, 200]

# ── Regime Detection ────────────────────────────────────────────────
REGIME_METHOD = "volatility"      # "volatility" or "hmm"
REGIME_LOOKBACK = 60              # days for regime classification
VOLATILITY_THRESHOLD_HIGH = 0.75  # percentile for "trending"
VOLATILITY_THRESHOLD_LOW = 0.25   # percentile for "mean-reverting"
HMM_N_STATES = 3                  # bull, bear, sideways

# ── Strategy Parameters ────────────────────────────────────────────
# Mean Reversion
MR_RSI_OVERSOLD = 30
MR_RSI_OVERBOUGHT = 70
MR_ZSCORE_ENTRY = -2.0           # buy when z-score below this
MR_ZSCORE_EXIT = 0.0             # exit when z-score reverts to 0
MR_BOLLINGER_ENTRY = -1.5        # standard deviations below mean

# Momentum
MOM_LOOKBACK = 30                 # primary momentum period
MOM_THRESHOLD = 0.05             # 5% return threshold to enter
MOM_TOP_N = 5                    # hold top N momentum assets
MOM_REBALANCE_DAYS = 7           # rebalance weekly

# Combined
REGIME_WEIGHT_MOMENTUM = 0.6     # weight in trending regime
REGIME_WEIGHT_MR = 0.4           # weight in mean-reverting regime
EQUAL_BLEND_WEIGHT = 0.5         # 50/50 blend as fallback

# ── Risk Management ─────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
KELLY_FRACTION = 0.25            # quarter-Kelly
MAX_POSITION_PCT = 0.15          # max 15% per position
MAX_PORTFOLIO_EXPOSURE = 0.90    # max 90% invested
STOP_LOSS_PCT = 0.05             # 5% stop-loss per position
TAKE_PROFIT_PCT = 0.15           # 15% take-profit per position
MAX_DRAWDOWN_HALT = 0.20         # halt trading at 20% drawdown
TRAILING_STOP_PCT = 0.08         # 8% trailing stop
TRANSACTION_COST_PCT = 0.001     # 0.1% per trade (maker fees)

# ── Backtesting ─────────────────────────────────────────────────────
WALK_FORWARD_TRAIN_DAYS = 365    # 1 year training window
WALK_FORWARD_TEST_DAYS = 90     # 3 month test window
WALK_FORWARD_STEP_DAYS = 30      # step forward 1 month
CPCV_N_SPLITS = 6                # combinatorial splits
CPCV_N_TEST_SPLITS = 2           # test groups per combination
PURGE_DAYS = 5                   # gap between train/test
MIN_TRADES_PER_FOLD = 20         # minimum trades for valid fold

# ── ML Models ───────────────────────────────────────────────────────
ENSEMBLE_MODELS = ["xgboost", "lightgbm", "ridge"]
ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]  # XGB, LGB, Ridge
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}
LGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

# ── Output ──────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
RESULTS_LOG = "outputs/results_log.csv"
