# Collin & Zog Wealth Management LLC — Trading Platform

## Overview

This repository contains two interconnected trading systems:

1. **Paper Trading Terminal (v4)** — A real-time paper trading simulator (`paper-trader-v4.html`)
2. **Quantitative Engine** — An institutional-grade backtesting engine with a web dashboard (`quant-engine/`)

The quant engine powers the analytical backend. It downloads real market data, runs six alpha models simultaneously, constructs optimized portfolios, enforces risk limits, and simulates execution with realistic transaction costs. Results are displayed in a live web dashboard styled to match the paper trading terminal.

---

## Quick Start

### Paper Trading Terminal
```
Open paper-trader-v4.html in any browser
```

### Quantitative Engine Dashboard
```bash
cd quant-engine
pip install -r requirements.txt
python server.py
# Open http://127.0.0.1:5000
```

Click **Run Backtest** in the dashboard to start. The engine downloads 5 years of data for 25 assets, runs all alpha models, and streams live results to the browser. A typical full run takes 60–90 seconds.

---

## What Does "Run Backtest" Actually Do?

When you press **Run Backtest**, the engine takes your $1,000,000 and simulates investing it over the past 5 years using real historical stock prices. It's a time machine for your money.

Here's what happens step by step:

1. **Downloads real market data** — 5 years of daily prices for 25 assets (stocks like AAPL, NVDA, TSLA, index funds like SPY and QQQ, bonds like TLT, commodities like GLD) from Yahoo Finance.

2. **Replays history day by day** — Starting from 5 years ago, the engine walks through every single trading day in order. On each day, it only knows what happened *up to that point* — no peeking at the future.

3. **Every 5 days, it decides what to buy and sell** — Six different strategy models each look at the data and say "I think AAPL will go up" or "I think TSLA will go down." Those opinions get blended together, then the engine figures out the best portfolio allocation given the risk constraints.

4. **It executes the trades with realistic costs** — Commissions, bid-ask spreads, and market impact are all deducted. This is critical because many strategies look profitable on paper but lose money once you account for trading fees.

5. **Risk limits are enforced continuously** — If the portfolio starts losing too much, the engine automatically reduces positions. If volatility spikes, it scales down. If any single stock gets too large, it trims it.

6. **The final number is what you'd have** — If your $1,000,000 turned into $1,003,753, the strategy made +0.4% over 5 years. That's not great — it basically broke even. If it turned into $1,200,000, that's +20% — much better.

**The point is to test before you risk real money.** If a strategy can't make money in a simulation with historical data, it definitely won't make money with real money. And if it *does* work in the simulation, it *might* work going forward (with the caveat that past performance doesn't guarantee future results).

### What If the Results Are Bad?

A flat or negative result with the default settings is actually useful information — it tells you that configuration doesn't work well. Try adjusting:

- **Fewer tickers** — Click Configure, enter `SPY,QQQ,AAPL,MSFT,NVDA` instead of 25 assets. Sometimes less is more.
- **Different portfolio method** — Switch from Risk Parity to Mean-Variance or Black-Litterman.
- **Toggle models on/off** — Maybe the ML model is adding noise. Try just momentum + mean reversion.
- **Shorter history** — 3 years instead of 5 captures more recent market behavior.
- **Higher vol target** — 15% or 20% takes more risk but can generate higher returns.

The goal is to find a configuration where the **Strategy Health Scorecard** (Analytics tab) lights up green.

---

## What the Dashboard Shows

### Dashboard Tab

**Hero Section** — The large number at the top is the portfolio's final (or current) equity value. Below it you'll see:

| Metric | What It Means |
|--------|---------------|
| **Sharpe Ratio** | Risk-adjusted return. Above 1.0 is good, above 2.0 is excellent. This is the single most important number — it tells you how much return you're getting per unit of risk. |
| **Ann. Return** | What the strategy would earn per year if performance continued at this rate. |
| **Ann. Vol** | How much the portfolio value fluctuates per year. Lower is calmer. The engine targets 10% by default. |
| **Max DD** | The worst peak-to-trough decline. This is your "worst day" number — how much you'd have lost if you started at the worst possible time. |
| **Win Rate** | Percentage of days with positive returns. Above 50% means more winning days than losing ones. |
| **Profit Factor** | Total gains divided by total losses. Above 1.5 means the strategy makes $1.50 for every $1 it loses. |

**Equity Curve** — The green line chart showing portfolio value over time. An upward-sloping line that doesn't have sharp drops is what you want. The chart updates live during the backtest so you can watch the simulation unfold.

**Current Positions** — Shows what the portfolio is holding right now and how much weight each asset gets. Green = long (betting it goes up), red = short (betting it goes down). The engine runs a long-short portfolio, meaning it can profit in both directions.

**Alpha Model Ensemble** — Shows how much weight each of the six alpha models is getting. The engine doesn't rely on a single strategy — it blends multiple signals. Models that produce more stable predictions get higher weight (inverse-volatility weighting). This is how institutional quant funds operate.

**Risk Dashboard** — Real-time risk metrics:
- **VaR (1d, 99%)** — The most you'd expect to lose in a single day, 99% of the time
- **Expected Shortfall** — Average loss on the worst 1% of days (the "tail risk")
- **Vol Scale** — How much the engine is scaling positions to hit the volatility target. Above 1.0 = levering up, below 1.0 = reducing exposure
- **Gross/Net Exposure** — Gross = total absolute position size, Net = long minus short. Low net exposure means the portfolio is close to market-neutral
- **HHI** — Concentration index. Lower = more diversified. 0.04 would be perfectly spread across 25 assets

**Recent Rebalances** — Log of when the portfolio was rebalanced, how many trades were executed, turnover percentage, and transaction costs.

### Analytics Tab

**Drawdown Chart** — Shows every peak-to-trough decline over the entire backtest. Deep red dips are the painful periods. The engine automatically reduces exposure when drawdown exceeds 7.5% and zeroes out positions at 15%.

**Return Distribution** — Histogram of daily returns. A healthy strategy has a slight right skew (more small gains than small losses) and thin tails (few extreme days). Heavy left tail = crash risk.

**Rolling 63-Day Sharpe** — Shows how the Sharpe ratio changes over time using a 3-month rolling window. Persistent positive values mean the strategy works across different market conditions. Periods where it drops below zero indicate regimes where the models struggled.

**Strategy Health Scorecard** — This evaluates the backtest against the targets defined in the trading research:

| Check | Target | Why It Matters |
|-------|--------|----------------|
| Win Rate >= 50% | Minimum viable | More winners than losers |
| Profit Factor >= 1.5 | Good quality | Wins meaningfully exceed losses |
| Sharpe >= 1.0 | Institutional grade | Consistent risk-adjusted performance |
| Max DD <= 20% | Capital preservation | Survivable worst case |
| Sortino >= 1.0 | Downside awareness | Penalizes only bad volatility |
| Calmar >= 1.0 | Recovery ability | Return relative to worst drawdown |
| Recovery Factor >= 2.0 | Resilience | Total return exceeds max drawdown by 2x |

A strategy that passes all checks is considered robust enough for live trading consideration (with small position sizes first).

### Trade Log Tab

Full history of every rebalance event with trade count, turnover, and costs. This is important because **transaction costs are the #1 killer of alpha** — a strategy that looks great on paper can lose money after costs. The engine models commissions (1 bps), spreads (2 bps), and market impact using the Almgren-Chriss square-root model.

### Research Notes Tab

Documents how findings from the deep research (`C:\Users\yarden\Desktop\trading-research`) were incorporated into the engine. Every strategy, indicator, risk technique, and backtesting practice from the five research documents is mapped to its implementation in the codebase.

---

## Why This Is Valuable

### It answers the question: "Would this have worked?"

Before risking real money, you need to know if a strategy actually generates returns after costs, across different market conditions, with proper risk management. This engine does exactly that — it simulates what would have happened if you ran this multi-model quant strategy over the past 5 years with $1M.

### It combines six independent alpha sources

No single strategy works all the time. The engine runs:

1. **Time-Series Momentum** — Trend following (works in trending markets)
2. **Cross-Sectional Momentum** — Long winners, short losers (captures relative strength)
3. **Momentum with Vol Break** — Reduces exposure during volatility spikes (avoids crashes)
4. **OU Mean Reversion** — Buys oversold assets, sells overbought (works in range-bound markets)
5. **Pairs Trading** — Trades cointegrated pairs when spreads deviate (market-neutral)
6. **ML Alpha (Gradient Boosting)** — Machine learning on 32+ features (captures nonlinear patterns)

By combining these, the portfolio performs more consistently than any single strategy alone.

### It enforces institutional-grade risk management

The engine implements the same risk controls used by hedge funds:
- **Volatility targeting** keeps risk constant across market regimes
- **VaR limits** prevent excessive tail risk
- **Drawdown protection** progressively de-risks during losing streaks
- **Position limits** prevent concentration in any single name
- **Transaction cost modeling** ensures results are realistic

---

## 24/7 Crypto Trading Mode

The paper trading terminal automatically switches to **crypto-only mode** when the stock market is closed (evenings, weekends, holidays). Crypto markets never sleep, so neither does the terminal.

### How It Works

- **During market hours (9:30 AM – 4:00 PM ET, Mon–Fri)** — The terminal trades all assets: stocks, ETFs, and crypto. Business as usual.
- **Outside market hours** — The terminal trades only crypto assets using live prices from the Binance public API. The status bar shows "24/7 Crypto Mode" so you always know what's happening.

### What Changes in Crypto Mode

| Feature | Normal Mode | Crypto Mode |
|---------|-------------|-------------|
| Tradeable assets | All (stocks + crypto) | Crypto only (14 assets) |
| Signal strength | Normal | Boosted 1.4x for crypto |
| Max positions | Standard limit | +6 additional slots |
| Position sizing | Normal | 1.3x larger for crypto |
| Trade labels | Standard | Tagged with `[24/7]` |

### Crypto Universe (14 assets)

BTC, ETH, SOL, ADA, AVAX, DOT, LINK — large-cap crypto
XRP, DOGE, POL, SUI, PEPE, NEAR, LTC — mid-cap and meme coins

All prices are fetched live from the Binance public API (`api.binance.com`). The quant engine also includes BTC-USD, ETH-USD, and SOL-USD in its backtest universe via Yahoo Finance.

### Why This Matters

Most retail trading tools go dark when the stock market closes. But crypto trades 24/7/365 — nights, weekends, Christmas. This feature means the terminal is always working, always generating signals, always looking for opportunities. When you wake up Monday morning, your crypto positions have been actively managed all weekend.

---

## Architecture

```
trading-bot/
├── paper-trader-v4.html          # Real-time paper trading terminal
├── README.md                     # This file
└── quant-engine/
    ├── server.py                 # Flask web server (API backend)
    ├── dashboard.html            # Web dashboard (v4-styled frontend)
    ├── main.py                   # CLI entry point (terminal dashboard)
    ├── config.py                 # All tunable parameters
    ├── requirements.txt          # Python dependencies
    ├── alpha/                    # Alpha signal generators
    │   ├── base.py               # Abstract base class
    │   ├── momentum.py           # TS momentum, XS momentum, vol break
    │   ├── mean_reversion.py     # OU mean reversion, pairs trading
    │   └── ml_alpha.py           # Gradient boosting ML model
    ├── backtest/
    │   └── engine.py             # Walk-forward backtesting engine
    ├── data/
    │   └── feed.py               # Market data from Yahoo Finance
    ├── ensemble/
    │   └── aggregator.py         # Signal combination (inverse vol, IC-weighted)
    ├── execution/
    │   └── engine.py             # Trade execution with cost modeling
    ├── features/
    │   └── engine.py             # 32+ feature engineering pipeline
    ├── portfolio/
    │   └── optimizer.py          # Mean-variance, risk parity, Black-Litterman
    ├── risk/
    │   └── manager.py            # VaR, drawdown, vol targeting, position limits
    └── ui/
        └── dashboard.py          # Rich terminal dashboard (CLI mode)
```

## Configuration

All parameters are in `quant-engine/config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tickers` | 28 diversified assets | Equity indices, bonds, commodities, mega-cap tech, financials, energy, crypto |
| `years` | 5 | Historical data lookback |
| `method` | risk_parity | Portfolio optimization (also: mean_variance, black_litterman) |
| `ensemble_method` | inverse_vol | Signal combination (also: equal_weight, performance_weighted) |
| `vol_target` | 10% | Annualized volatility target |
| `max_drawdown` | 15% | Hard stop — zeroes all positions if breached |
| `max_position_size` | 15% | Maximum weight in any single asset |
| `rebalance_frequency` | 5 days | How often the portfolio rebalances |

These can all be adjusted from the web dashboard's Configure panel before running a backtest.

---

## Change Log

### v5.0 — Adaptive Learning Brain (2026-03-10)

**Feature:** The paper trading terminal now has a full machine learning system that gets smarter the longer it runs. The bot tracks which strategies win and lose, adapts its behavior, detects market regimes, and persists all learning across browser sessions.

**1. Strategy Performance Tracking**
- Every completed trade feeds back into the learning system via `updateLearning()`
- Per-strategy EMA-smoothed win rate and average PnL (decay factor alpha=0.15, ~12-15 trade effective memory)
- Tracks last 20 results per strategy for recent performance context
- Only entry strategies are learned from: RSI, MA, Momentum, MeanRev, VWAP, Composite

**2. Adaptive Signal Weighting**
- Each strategy gets a `weight` multiplier (range 0.2x to 3.0x) applied to its signal strength
- Weights recalculate after every trade using: 60% win rate score + 40% PnL direction score
- Winning strategies get boosted (e.g., RSI at 100% WR → 2.29x), losing ones get suppressed (e.g., MeanRev at 0% WR → 0.45x)
- Minimum 5 trades required before weights adjust (prevents wild swings on small samples)
- Weight formula: `exp(combined * 1.1)` with `combined = wrScore*0.6 + pnlScore*0.4`

**3. Dynamic Threshold Adjustment**
- Tracks win/loss streaks across all strategies
- On losing streak (-3): SL tightens 30%, TP drops 15%, entry bar raises 30%
- On losing streak (-5): SL tightens 50%, TP drops 30%, entry bar raises 60%
- On winning streak (+3): SL widens 15%, TP raises 20%, entry bar lowers 15%
- Between streaks: thresholds gradually decay back toward 1.0x (30% per trade)

**4. Regime Detection**
- Analyzes price history across all assets every tick
- Measures directional consistency (consecutive same-sign returns) and return volatility
- Classifies market as: TRENDING (consistency > 55%), CHOPPY (high vol + low consistency), or MIXED
- Regime adjusts strategy preferences:
  - Trending: boosts MA (1.2x), Momentum (1.2x); suppresses MeanRev (0.7x)
  - Choppy: boosts MeanRev (1.25x), RSI (1.2x); suppresses Momentum (0.7x)
- Requires 40% confidence minimum before applying regime adjustments

**5. Data Persistence**
- All learning data auto-saved to `localStorage` (key: `czwm_brain_v1`) after every completed trade
- On page load, `loadBrain()` restores all previous session data automatically
- Brain data is ~1.7KB — small enough for unlimited saves
- **Export Brain**: Downloads all learning data as `czwm-brain-YYYY-MM-DD.json` file
- **Import Brain**: Uploads a previously exported JSON file to restore learning state
- **Reset Learning**: Wipes all learned data while preserving trading positions and cash

**6. Visible UI Panel ("Adaptive Brain" card)**
- Strategy Weights: color-coded weight bars (green = boosted, red = suppressed, gray = neutral)
- Scorecard: win rate, average PnL, and trade count per strategy, plus current streak (green for wins, red for losses)
- Regime & Thresholds: current detected regime (TRENDING/CHOPPY/MIXED) with confidence %, SL/TP/Entry multipliers, total learning trades
- Meta bar: "Last saved" timestamp and localStorage key
- Learning toggle: on/off switch to compare performance with and without adaptive learning

**Verified with tests:** 9/9 tests passed — weights adapt correctly (5x difference between winning and losing strategies), thresholds adjust on streaks, regime adjustments work, brain serializes under 5KB.

---

### v4.5 — Fix TP/SL Thresholds for Tick-Level Trading (2026-03-10)

**Problem:** Trades were executing (buys), but positions never closed for meaningful profit or loss. The take-profit (50%), stop-loss (-15%), and trailing stop (10%) thresholds were calibrated for daily/weekly holding periods. With tick-level data where crypto moves ~0.1-0.5% over minutes, positions would need to be held for hours or days before any exit trigger fired. The PnL column showed "-" (buys) or "$0.00" because nothing was ever sold.

**What the portfolio display means:** When it shows "$280 cash / $720 crypto", that means $720 is currently invested in open crypto positions and $280 is uninvested. The total is still ~$1,000. Cash decreases on buys and increases on sells.

**Changes to `paper-trader-v4.html`:**

- **Take-profit thresholds** reduced to match tick-level price movements:
  - Moderate: 20% → 1.2%
  - Aggressive: 35% → 2.0%
  - YOLO: 50% → 3.0%
- **Stop-loss thresholds** tightened:
  - Moderate: -6% → -0.8%
  - Aggressive: -10% → -1.2%
  - YOLO: -15% → -1.8%
- **Trailing stop** tightened:
  - Moderate: 4% → 0.5%
  - Aggressive: 6% → 0.8%
  - YOLO: 10% → 1.0%
- **Trailing stop activation**: Lowered from 1.5% profit minimum to 0.1%, so the trail kicks in as soon as there's any meaningful gain
- **Options TP/SL**: Take profit 50% → 15%, stop loss -60% → -20%

**Verified with simulation test:** All 3 exit types confirmed working:
- Take Profit: BTC +3.0% → sold for +$18.00
- Stop Loss: ETH -1.8% → sold for -$9.00
- Trailing Stop: SOL peaked +1.16%, dropped 1% from high → sold for +$0.47

---

### v4.4 — Fix Crypto Price Source (Binance US + CoinGecko Fallback) (2026-03-10)

**Problem:** Zero trades were happening because the Binance API (`api.binance.com`) returns **HTTP 451 (Unavailable For Legal Reasons)** — it's geo-blocked in the United States. Every crypto price fetch silently failed (caught by try/catch), so `curPrices` stayed empty and no signals could ever fire. This was the root cause of no trades.

**Changes to `paper-trader-v4.html`:**

- Replaced `api.binance.com` with **`api.binance.us`** (Binance US) as primary crypto price source
- Added **CoinGecko** (`api.coingecko.com`) as automatic fallback if Binance US fails
- Refactored `fetchCrypto()` into three functions: `fetchCrypto()` (orchestrator), `_fetchBinance()` (Binance US), `_fetchCoinGecko()` (fallback), `_applyCryptoPrices()` (shared price storage)
- Both APIs tested and confirmed working from US IP

**Verified with end-to-end test:**
- Fetched 15 ticks of live prices from Binance US (all 14 crypto assets loaded)
- 3 trades executed automatically: BTC (Momentum +0.208%), LINK (MeanRev Z=-1.00), XRP (Composite 0.300)
- Portfolio fully invested ($1000 → 3 positions) within ~12 seconds

---

### v4.3 — Signal Tuning for Live Tick Data (2026-03-10)

**Problem:** No trades were executing in the paper trading terminal. The trading signals were calibrated for daily price bars (large price swings over 24 hours), but the terminal feeds live tick data at 6–10 second intervals where prices barely move between samples. RSI stayed near 50, momentum stayed near 0, and no signals ever fired.

**Changes to `paper-trader-v4.html`:**

- **Shortened indicator periods** for faster warmup with tick data:
  - Moving averages: 4/12 → 3/8 (crossovers happen in ~48s instead of ~2 min)
  - Momentum lookback: 8 → 5 ticks
  - RSI period: 14 → 8
  - Z-score window: 15 → 8
  - VWAP period: 10 → 6
  - MIN_HIST: 5 → 3
- **Lowered signal thresholds** to match tick-level price movements:
  - Momentum thresholds: 0.4%–1.2% → 0.1%–0.3% (per risk level)
  - VWAP entry: 1.5% below → 0.3% below
  - Composite score threshold: 0.15 → 0.06
  - Mean reversion Z-threshold: 0.8 → 0.5
  - Options composite thresholds: 0.1 → 0.04
- **Increased composite sub-signal weights** so each indicator contributes more to the overall score (RSI: 0.3→0.4, MA: 0.25→0.3, Momentum: 0.3→0.35, etc.)
- **Added MicroTrend signal** — detects 3+ consecutive ticks moving in the same direction with >0.05% cumulative move. Purpose-built for tick-level data where traditional indicators are too slow.
- **Faster refresh**: Default changed from 10s to 6s, added 3s option. Finnhub stock throttle reduced from 15s to 8s.

**Result:** Trades should begin firing within 30–60 seconds of pressing START as the bot accumulates enough tick history for signals to trigger.

---

### v4.2 — 24/7 Crypto Trading Mode (2026-03-10)

**Changes to `paper-trader-v4.html`:**

- Expanded crypto universe from 7 to 14 assets (added XRP, DOGE, POL, SUI, PEPE, NEAR, LTC)
- Added automatic crypto-only mode when US stock market is closed
- Crypto signal strength boosted 1.4x during off-hours
- Max positions increased by 6 during crypto mode
- Crypto position sizing boosted 1.3x during crypto mode
- Trade reasons tagged with `[24/7]` in crypto-only mode
- Hero section shows separate crypto portfolio value
- Status bar displays "24/7 Crypto Mode" with context (Weekend/Pre-market/After hours)
- Updated brand text to "24/7 Options, Equities & Crypto Terminal v4"

**Changes to `quant-engine/config.py`:**

- Added BTC-USD, ETH-USD, SOL-USD to default backtest universe

---

### v4.1 — Quant Engine Dashboard & Bug Fixes (2026-03-10)

**Problem:** Quant engine crashed/hung on startup. ML Alpha model was too slow with 25-ticker universe (200 GBM estimators retrained every 63 days across 25 assets = infinite hang).

**Fixes:**

- `quant-engine/alpha/ml_alpha.py` — Scaled ML complexity with universe size: retrain frequency = max(config, n_assets*5), estimators capped at 100 for 10+ assets and 50 for 20+ assets. Full backtest went from infinite hang to ~66 seconds.
- `quant-engine/main.py` — Fixed `plt.show()` blocking with `block=False` + `pause(0.5)` + `close("all")`

**New files:**

- `quant-engine/server.py` — Flask web server with REST API (`/api/run`, `/api/status`, `/api/result`, `/api/defaults`). Background threading, NumpyEncoder for JSON serialization, live progress streaming via WebDashboardCallback.
- `quant-engine/dashboard.html` — Full web dashboard styled to match paper-trader-v4. Four tabs (Dashboard, Analytics, Trade Log, Research Notes). Live equity curve, drawdown chart, return distribution, rolling Sharpe, strategy health scorecard. Configuration panel for all parameters.
- `quant-engine/requirements.txt` — Added flask>=3.0.0, flask-cors>=4.0.0
- `trading-bot/README.md` — This documentation file

**Research incorporated** from `C:\Users\yarden\Desktop\trading-research\` (5 documents covering core strategies, ML approaches, risk management, technical indicators, and backtesting frameworks). All findings mapped to implementation in the Research Notes tab.
