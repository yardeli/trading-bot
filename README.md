# Quant Trading Platform

A quantitative trading system with 6 alpha models, adaptive ensemble weighting, and live paper trading via Alpaca Markets.

## Systems

| System | What it does | How to run |
|---|---|---|
| **Unified Dashboard** | One-click backtest + live trading | `python dashboard.py` |
| **Live Trader** | Autonomous Alpaca paper trading | `python live_trader.py` |
| **Signal Server** | REST API serving quant signals | `python signal_server.py` |
| **Paper Trading Terminal** | Browser-based trading simulator | Open `paper-trader-v4.html` |
| **Daily Backtester** | Intraday strategy backtester | Open `daily-backtester.html` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the unified dashboard
python dashboard.py

# Open http://localhost:8888
# Pick Backtest or Live Trading mode
# Press START
```

## Unified Dashboard (`dashboard.py`)

The main interface. Open `http://localhost:8888` and choose your mode:

### Backtest Mode
- Press **START** to run the quant engine on 5 years of historical data
- Watch the equity curve build in real time
- See performance metrics: Sharpe, Sortino, Calmar, drawdown, win rate
- Review every rebalance in the Trade Log tab
- Current performance: **7.41% annualized return, 0.49 Sharpe, -19.65% max drawdown**

### Live Trading Mode
1. Click **Live Trading** in the mode toggle
2. Select rebalance interval (2 min to 1 hour)
3. Press **START**
4. The system connects to Alpaca and begins trading:
   - Downloads historical data for model warmup
   - Generates signals from 6 alpha models
   - Optimizes portfolio via mean-variance
   - Executes paper trades on Alpaca
   - Auto-rebalances at your chosen interval
5. Use **REBALANCE NOW** to force an immediate cycle
6. Use **STOP** to halt trading

### Dashboard Features
- **Equity curve** — live-streaming chart (green for backtest, orange for live)
- **Drawdown chart** — peak-to-trough visualization
- **Performance metrics** — return, Sharpe, Sortino, Calmar, win rate, profit factor
- **Model weights** — which of the 6 alpha models are driving trades
- **Trade log** — every trade with date, symbol, side, amount, status
- **Positions tab** — live Alpaca positions with P&L
- **Weights & Risk tab** — portfolio weights and risk metrics (VaR, exposure, concentration)

## Quant Engine

The engine lives in `../quant-engine-local/` and powers all signal generation. See [quant-engine README](https://github.com/yardeli/quant-engine) for full docs.

### 6 Alpha Models

| Model | Type | Strategy |
|---|---|---|
| **TS Momentum** | Trend following | Continuous z-score signals with vol scaling |
| **XS Momentum** | Cross-sectional | 12-1 month return, rank-normalized |
| **Vol Brake Momentum** | Crisis protection | Momentum + vol spike + correlation crash filter |
| **OU Mean Reversion** | Mean reversion | Ornstein-Uhlenbeck with adaptive half-life |
| **Pairs Trading** | Statistical arbitrage | Engle-Granger cointegration, top 5 pairs |
| **ML Alpha** | Machine learning | Gradient boosting on 20+ features |

### Pipeline

```
Market Data (yfinance) → Features (20+) → 6 Alpha Models → Ensemble (IC-weighted)
→ Portfolio Optimizer (mean-variance) → Risk Manager (VaR, vol targeting, DD limits)
→ Execution (commission + spread + market impact)
```

## Performance (Backtest)

| Metric | Value |
|---|---|
| Annualized Return | 7.41% |
| Sharpe Ratio | 0.49 |
| Sortino Ratio | 0.63 |
| Max Drawdown | -19.65% |
| Total Return (5yr) | 47.89% |
| Profit Factor | 1.11 |

## Alpaca Configuration

The system uses Alpaca paper trading. Set your credentials via environment variables:

```bash
export ALPACA_KEY=your_key_here
export ALPACA_SECRET=your_secret_here
```

Or they default to the paper trading credentials in the code.

## Project Structure

```
trading-bot/
├── dashboard.py           # Unified dashboard (backtest + live)
├── live_trader.py         # Autonomous live trading bot
├── signal_server.py       # REST API signal server
├── paper-trader-v4.html   # Browser-based paper trading terminal
├── daily-backtester.html  # Intraday strategy backtester
├── index.html             # Navigation page
├── requirements.txt       # Python dependencies
└── brain_export.json      # Exported model weights

quant-engine-local/
├── alpha/                 # 6 alpha models
├── backtest/              # Walk-forward backtester
├── data/                  # yfinance data loader
├── ensemble/              # Signal aggregation
├── execution/             # Transaction cost modeling
├── features/              # 20+ technical features
├── portfolio/             # Mean-variance, risk parity, Black-Litterman
├── risk/                  # VaR, vol targeting, drawdown limits
├── config.py              # All tunable parameters
├── main.py                # CLI entry point
└── server.py              # Flask web server
```

## Dependencies

```
alpaca-py>=0.30.0
flask>=3.0.0
flask-cors>=4.0.0
yfinance>=0.2.31
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
```
