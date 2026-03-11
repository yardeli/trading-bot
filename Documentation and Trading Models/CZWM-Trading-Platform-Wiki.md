# CZWM Trading Platform — Comprehensive Wiki

> **Collin & Zog Wealth Management LLC**
> Last Updated: March 10, 2026
> Version: 5.0 (Adaptive Learning Brain)

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [System Architecture](#2-system-architecture)
3. [Paper Trading Terminal (v4/v5)](#3-paper-trading-terminal)
4. [Quantitative Engine](#4-quantitative-engine)
5. [Core Trading Strategies](#5-core-trading-strategies)
6. [Technical Indicators Reference](#6-technical-indicators-reference)
7. [Machine Learning & AI](#7-machine-learning--ai)
8. [Adaptive Learning Brain (v5.0)](#8-adaptive-learning-brain)
9. [Risk Management Framework](#9-risk-management-framework)
10. [Regime Detection](#10-regime-detection)
11. [Position Sizing & Kelly Criterion](#11-position-sizing--kelly-criterion)
12. [Backtesting Best Practices](#12-backtesting-best-practices)
13. [Performance Metrics Glossary](#13-performance-metrics-glossary)
14. [Planned Enhancements (ML Expansion)](#14-planned-enhancements)
15. [Configuration Reference](#15-configuration-reference)
16. [Change Log](#16-change-log)

---

## 1. Platform Overview

The CZWM Trading Platform consists of two interconnected systems that together provide a complete quantitative trading workflow — from research and backtesting to live paper trading with adaptive learning.

**Paper Trading Terminal** — A single-file HTML application (`paper-trader-v4.html`) that runs entirely in the browser with no backend. It fetches live prices, generates trading signals from 6+ strategies, executes paper trades, and now includes an adaptive learning system that improves the longer it runs. It trades stocks, ETFs, options, and 14 crypto assets 24/7.

**Quantitative Engine** — A Python-based institutional-grade backtesting engine with a web dashboard. It downloads 5 years of real market data for 25+ assets, runs 6 alpha models simultaneously, constructs optimized portfolios, enforces risk limits, and simulates execution with realistic transaction costs. A typical full run takes 60–90 seconds.

**How They Connect:** The quant engine is the research lab — you test strategies on historical data to see what would have worked. The paper trader is the live field test — it applies those strategies to real-time market data. The adaptive learning system (v5.0) bridges the gap by letting the paper trader learn from its own live results and adjust in real time.

---

## 2. System Architecture

```
trading-bot/
├── paper-trader-v4.html            # Real-time paper trading terminal (browser-only)
├── README.md                       # Change log and documentation
└── quant-engine/
    ├── server.py                   # Flask web server (API backend)
    ├── dashboard.html              # Web dashboard (v4-styled frontend)
    ├── main.py                     # CLI entry point (terminal dashboard)
    ├── config.py                   # All tunable parameters
    ├── requirements.txt            # Python dependencies
    ├── alpha/                      # Alpha signal generators
    │   ├── base.py                 # Abstract base class
    │   ├── momentum.py             # TS momentum, XS momentum, vol break
    │   ├── mean_reversion.py       # OU mean reversion, pairs trading
    │   └── ml_alpha.py             # Gradient boosting ML model (32+ features)
    ├── backtest/engine.py          # Walk-forward backtesting engine
    ├── data/feed.py                # Market data from Yahoo Finance
    ├── ensemble/aggregator.py      # Signal combination (inverse vol, IC-weighted)
    ├── execution/engine.py         # Trade execution with cost modeling
    ├── features/engine.py          # 32+ feature engineering pipeline
    ├── portfolio/optimizer.py      # Mean-variance, risk parity, Black-Litterman
    ├── risk/manager.py             # VaR, drawdown, vol targeting, position limits
    └── ui/dashboard.py             # Rich terminal dashboard (CLI mode)
```

**Data Flow:**

```
Paper Trader:
  Live Price APIs (Binance US, CoinGecko, Finnhub)
    → Price History Buffer (in-memory)
    → Technical Indicator Calculations
    → 6 Signal Generators (RSI, MA, Momentum, MeanRev, VWAP, Composite)
    → Adaptive Learning Weights (v5.0)
    → Regime Detection Filter
    → Position Sizing / Risk Check
    → Trade Execution
    → Performance Tracking → Learning Feedback Loop
    → localStorage Persistence

Quant Engine:
  Yahoo Finance (5yr historical data)
    → Feature Engineering (32+ features)
    → 6 Alpha Models (parallel)
    → Ensemble Aggregator (inverse vol weighting)
    → Portfolio Optimizer (risk parity / mean-variance / Black-Litterman)
    → Risk Manager (VaR, drawdown, vol targeting, position limits)
    → Execution Engine (commission, spread, market impact modeling)
    → Performance Analytics → Web Dashboard
```

---

## 3. Paper Trading Terminal

### What It Does

The paper trader is your live testing ground. It connects to real price feeds, runs your strategies against live tick data, and manages a simulated portfolio. With v5.0, it also learns from every trade it makes.

### Asset Universe

**Stocks & ETFs (during market hours 9:30 AM – 4:00 PM ET):** Any asset tradeable through the configured data sources.

**Crypto (24/7, 14 assets):** BTC, ETH, SOL, ADA, AVAX, DOT, LINK, XRP, DOGE, POL, SUI, PEPE, NEAR, LTC

When the US stock market closes, the terminal automatically switches to **crypto-only mode**. Crypto signals get a 1.4x strength boost, max positions increase by 6, and position sizing gets a 1.3x multiplier. Trades during crypto-only mode are tagged `[24/7]`.

### Price Sources

Primary crypto source is **Binance US** (`api.binance.us`), with **CoinGecko** as automatic fallback. The original `api.binance.com` endpoint is geo-blocked in the US (HTTP 451). Stock data comes from **Finnhub** with an 8-second throttle.

### Signal Types

The paper trader generates signals from 6 independent strategies:

| Signal | Type | When It Fires |
|--------|------|---------------|
| **RSI** | Momentum / Mean Reversion | RSI crosses below oversold (buy) or above overbought (sell) threshold |
| **MA Crossover** | Trend Following | Fast moving average (3-period) crosses above/below slow (8-period) |
| **Momentum** | Trend Following | Price change over 5-tick lookback exceeds threshold (0.1%–0.3% depending on risk level) |
| **Mean Reversion** | Mean Reversion | Z-score of price vs. 8-tick rolling mean exceeds ±0.5 |
| **VWAP** | Institutional | Price deviates 0.3%+ below volume-weighted average price |
| **Composite** | Ensemble | Weighted combination of RSI (0.4), MA (0.3), Momentum (0.35) scores exceeds 0.06 threshold |
| **MicroTrend** | Tick-Level | 3+ consecutive ticks in same direction with >0.05% cumulative move |

All indicator periods are calibrated for tick-level data (6–10 second intervals), not daily bars. This was addressed in v4.3.

### Exit Conditions (Tick-Level Calibrated)

Thresholds were recalibrated in v4.5 for tick-level trading where crypto typically moves 0.1%–0.5% over minutes:

| Parameter | Moderate | Aggressive | YOLO |
|-----------|----------|------------|------|
| Take Profit | 1.2% | 2.0% | 3.0% |
| Stop Loss | -0.8% | -1.2% | -1.8% |
| Trailing Stop | 0.5% | 0.8% | 1.0% |
| Trailing Activation | 0.1% profit minimum | | |
| Options TP / SL | 15% / -20% | | |

### Portfolio Display

When the terminal shows "$280 cash / $720 crypto," it means $720 is currently invested in open positions and $280 is uninvested. The total remains ~$1,000. Cash decreases on buys and increases on sells.

---

## 4. Quantitative Engine

### What "Run Backtest" Does

When you press Run Backtest, the engine takes $1,000,000 and simulates investing it over the past 5 years using real historical prices. Step by step:

1. **Downloads real market data** — 5 years of daily prices for 25+ assets from Yahoo Finance
2. **Replays history day by day** — Only uses data available up to each date (no future peeking)
3. **Every 5 days, rebalances** — 6 alpha models each generate signals, blended together, then portfolio is optimized
4. **Executes with realistic costs** — Commissions (1 bps), spreads (2 bps), market impact (Almgren-Chriss model)
5. **Enforces risk limits continuously** — Drawdown protection, volatility targeting, position limits
6. **Reports final result** — What your $1M turned into after 5 years

### Six Alpha Models

| # | Model | Strategy Type | When It Works |
|---|-------|---------------|---------------|
| 1 | Time-Series Momentum | Trend Following | Trending markets (bull or bear) |
| 2 | Cross-Sectional Momentum | Relative Strength | Long winners, short losers |
| 3 | Momentum with Vol Break | Trend + Risk Control | Reduces exposure during volatility spikes |
| 4 | OU Mean Reversion | Mean Reversion | Range-bound / choppy markets |
| 5 | Pairs Trading | Statistical Arbitrage | Cointegrated pairs diverge then converge |
| 6 | ML Alpha (Gradient Boosting) | Machine Learning | Captures nonlinear patterns from 32+ features |

Models are combined using **inverse-volatility weighting** — models with more stable predictions get higher weight. This is how institutional quant funds operate.

### Dashboard Tabs

**Dashboard:** Hero metrics (Sharpe, annual return, volatility, max drawdown, win rate, profit factor), live equity curve, current positions, alpha model ensemble weights, risk dashboard (VaR, expected shortfall, vol scale, gross/net exposure, HHI concentration).

**Analytics:** Drawdown chart, return distribution histogram, rolling 63-day Sharpe, Strategy Health Scorecard (7 checks: win rate ≥50%, profit factor ≥1.5, Sharpe ≥1.0, max DD ≤20%, Sortino ≥1.0, Calmar ≥1.0, recovery factor ≥2.0).

**Trade Log:** Full rebalance history with trade count, turnover, and costs.

**Research Notes:** Maps all trading research findings to their implementation in the codebase.

### Tuning Bad Results

If a backtest produces flat or negative results, try: fewer tickers (e.g., just `SPY,QQQ,AAPL,MSFT,NVDA`), a different portfolio method (switch from Risk Parity to Mean-Variance), toggling specific models on/off, shorter history (3 years instead of 5), or higher volatility target (15%–20%).

---

## 5. Core Trading Strategies

### Trend Following

Identifies and follows market trends. Uses moving averages (20/50/200-day), momentum oscillators, price breakouts. Works best in trending markets. Entry when price breaks above MA, exit when trend reverses. Risk: false breakouts in choppy markets.

### Mean Reversion

Profits from temporary price deviations reverting to their mean. Uses Bollinger Bands, RSI extremes (<30 oversold, >70 overbought), Keltner Channels. Works best in range-bound / sideways markets. Entry at extremes, exit at mean. Risk: strong trends overwhelm reversion.

### Momentum

Capitalizes on continuation of existing trends. Uses MACD, RSI confirmation, Stochastic Oscillator, volume confirmation. Entry when multiple MAs align and MACD crosses signal line. Risk: late entry where momentum is already exhausted, whipsaws in choppy conditions.

### Statistical Arbitrage (Pairs Trading)

Exploits temporary pricing inefficiencies between correlated assets. Find pairs with correlation >0.8, track their price ratio, enter when z-score exceeds ±2.5, exit when it returns to ±1.0. Advantage: market-neutral (profits from relative mispricing, not market direction). The quant engine uses **cointegration** (Engle-Granger test) rather than simple correlation — cointegration guarantees mathematical convergence, which is more robust.

### Momentum Breakout

Identifies strong trends early via breakouts from consolidation patterns confirmed by volume spikes. Best at the beginning of major trends.

### Factor Investing

Builds portfolios tilted toward specific return-driving factors:
- **Value:** Low P/E, high dividend yield (4–5% annual premium historically)
- **Momentum:** Recent strong performers (6–8% annual premium)
- **Size:** Small-cap outperformance (2–3% premium, inconsistent)
- **Quality:** High ROE, strong balance sheets (3–4% premium)
- **Low Volatility:** Lower-vol stocks with same/better returns (works well during stress)

Multi-factor approach scores each stock across multiple dimensions and goes long top quartile, short bottom quartile.

### Dynamic Regime-Aware Trading

Switches strategies based on detected market regime. This is one of the single most impactful improvements — no single strategy works in all conditions. See [Section 10: Regime Detection](#10-regime-detection) for full details.

### Market-Neutral Long/Short

Long high-quality stocks, short poor-quality stocks, with equal dollar exposure on both sides. Net market exposure ≈ 0%. Profit comes purely from security selection (alpha), not market direction. The quant engine runs a long-short portfolio.

### Volatility-Based Trading

Use volatility as a signal rather than something to trade directly. When VIX is high (>30): tighter stops, smaller positions, mean reversion preferred. When VIX is low (<15): wider stops, larger positions, trend following preferred. Mathematical approach: position size = base_size × (1 / (VIX / 20)).

---

## 6. Technical Indicators Reference

### Trend Indicators

**Moving Averages (MA):** Smooth price data to show trend direction. Paper trader uses 3/8 periods (tick-calibrated), quant engine uses 20/50/200 (daily). Stacked MAs (price > short > medium > long) = strong uptrend.

**MACD (12/26/9):** Shows relationship between two EMAs. Buy when MACD crosses above signal line. Good for trend confirmation but slower than price action.

**ADX (Average Directional Index):** Measures trend *strength*, not direction. ADX <20 = weak/no trend (avoid trend-following), ADX 20–40 = moderate trend, ADX >40 = strong trend. Critical for regime detection.

### Momentum Indicators

**RSI (Relative Strength Index):** Range 0–100. Below 30 = oversold, above 70 = overbought. Paper trader uses 8-period (tick-calibrated). In trend-following mode, RSI >50 confirms bullish momentum.

**Stochastic Oscillator:** Compares close to price range. More sensitive to recent price action than RSI. Below 20 = oversold, above 80 = overbought.

**ROC (Rate of Change):** Speed of price movement. Confirm breakouts by checking ROC magnitude.

### Volatility Indicators

**Bollinger Bands (20, 2σ):** Price envelope based on volatility. Price touching lower band + RSI oversold = strong mean reversion signal. Band squeeze (very narrow) often precedes big breakout.

**ATR (Average True Range, 14-period):** Measures daily volatility. Used for stop-loss placement (stop = entry − 2×ATR), position sizing (risk per share = 1–2×ATR), and trade filtering (skip low-ATR assets with insufficient movement).

**VIX:** Market fear gauge. <15 = calm, 15–20 = normal, 20–30 = elevated, >30 = extreme fear. Used for regime detection and position sizing.

### Volume Indicators

**Volume:** Confirms conviction. High-volume breakout = strong signal. Low-volume breakout = weak (likely to fail). Divergence (price rising but volume falling) = warning.

**VWAP:** Volume-weighted average price. Institutional benchmark. Price above VWAP = institutions buying. Paper trader triggers signals when price deviates 0.3%+ below VWAP.

**OBV (On Balance Volume):** Cumulative volume indicator. Rising OBV = volume supports uptrend. Divergence between OBV and price = warning of reversal.

### Recommended Combinations

| Market Condition | Indicators | Strategy |
|------------------|-----------|----------|
| Strong Uptrend | MA aligned, MACD positive, RSI >50, ADX >25 | Momentum / trend following |
| Strong Downtrend | MA reversed, MACD negative, RSI <50, ADX >25 | Trend following (short) |
| Range-Bound | Bollinger Bands, RSI extremes, low ADX | Mean reversion |
| High Volatility | ATR expanding, VIX >25 | Wider stops, smaller positions |
| Breakout Setup | Volume spike, MA break, ATR expanding | Momentum |

---

## 7. Machine Learning & AI

### Current Implementation (Quant Engine)

The quant engine includes a **Gradient Boosting (GBM) ML Alpha model** that trains on 32+ engineered features from historical data. It runs only during backtests — it does not feed into the live paper trader. ML complexity scales with universe size: retrain frequency = max(config, n_assets×5), estimators capped at 100 for 10+ assets and 50 for 20+.

### Neural Network Architectures (Research Reference)

**LSTM (Long Short-Term Memory):** Best for time-series prediction. Captures long-term dependencies. Input: 60 days OHLCV → multiple LSTM layers → dense layers → direction prediction. Reported accuracies up to 93% in research (though real-world is more like 55–60% directional accuracy).

**Transformers:** Better than LSTM for long-range dependencies via attention mechanism. Parallel processing = faster training. State-of-the-art for time series. Most complex to implement.

**CNN (Convolutional Neural Networks):** Treat candlestick charts as images, extract spatial patterns. Can identify head-and-shoulders, triangles, etc. Interesting but less proven for scalping.

**Ensemble Methods:** Combine multiple models, average predictions. More robust, reduces overfitting. The quant engine already does this with its 6 alpha models.

### Reinforcement Learning (Research Reference)

RL agents learn trading strategy through trial and error: observe market state → take action (buy/sell/hold) → receive reward (profit/loss) → update policy. Key algorithms:
- **Deep Q-Learning (DQL):** Neural network approximates action values. Discrete actions only.
- **Policy Gradient (REINFORCE):** Learns action probabilities directly. Can handle continuous position sizing. High variance.
- **PPO (Proximal Policy Optimization):** Recommended. Stable, fast convergence, clips updates to prevent overshooting.
- **Actor-Critic:** Two networks (actor = policy, critic = value). Lower variance than pure policy gradient. Best for complex trading.

### Sentiment Analysis (Research Reference)

Three tiers: lexicon-based (simple word scoring, fast, limited), LSTM-based (learns context from training data), transformer-based (BERT/GPT, best accuracy, pre-trained, but heavy). Integration: sentiment score combined with technical signal using weighted addition. Both must agree for high-confidence trades.

### Hybrid Approach (Recommended)

Pure ML overfits and breaks on regime changes. Pure rules miss complex patterns. Best approach: ML finds hidden patterns while rules provide guardrails. The paper trader's adaptive learning system (v5.0) is a step toward this — it uses rule-based signals but adapts their weights based on observed performance.

---

## 8. Adaptive Learning Brain (v5.0)

This is the paper trader's learning system, added in v5.0. It makes the bot genuinely smarter the longer it runs.

### 1. Strategy Performance Tracking

Every completed trade feeds back into the learning system via `updateLearning()`. Per-strategy metrics are tracked using EMA (exponential moving average) smoothing with alpha = 0.15, giving an effective memory of about 12–15 trades. Tracks: win rate, average PnL, trade count, and last 20 results. Only entry strategies are learned from: RSI, MA, Momentum, MeanRev, VWAP, Composite.

### 2. Adaptive Signal Weighting

Each strategy gets a weight multiplier (range 0.2x to 3.0x) applied to its signal strength. Weights recalculate after every trade using: 60% win rate score + 40% PnL direction score. Formula: `weight = exp(combined × 1.1)`. Example: RSI at 100% win rate → 2.29x boost, MeanRev at 0% win rate → 0.45x suppression. Minimum 5 trades required before weights adjust (prevents swinging on small samples).

### 3. Dynamic Threshold Adjustment

Tracks win/loss streaks across all strategies and adjusts entry/exit thresholds:

| Streak | Stop-Loss | Take-Profit | Entry Bar |
|--------|-----------|-------------|-----------|
| -3 losing | Tighten 30% | Drop 15% | Raise 30% |
| -5 losing | Tighten 50% | Drop 30% | Raise 60% |
| +3 winning | Widen 15% | Raise 20% | Lower 15% |
| Between streaks | Decay toward 1.0x (30% per trade) | | |

### 4. Regime Detection

Analyzes price history across all assets every tick. Measures directional consistency (consecutive same-sign returns) and return volatility. Classification:

| Regime | Condition | Strategy Adjustments |
|--------|-----------|---------------------|
| TRENDING | Consistency >55% | Boost MA (1.2x), Momentum (1.2x); suppress MeanRev (0.7x) |
| CHOPPY | High vol + low consistency | Boost MeanRev (1.25x), RSI (1.2x); suppress Momentum (0.7x) |
| MIXED | Everything else | No adjustments |

Requires 40% confidence minimum before applying.

### 5. Data Persistence

All learning data auto-saves to `localStorage` (key: `czwm_brain_v1`) after every completed trade. On page load, `loadBrain()` restores everything. Brain data is ~1.7KB. Export Brain downloads a JSON backup file, Import Brain restores from a previous export, Reset Learning wipes learned data while preserving positions.

### 6. UI Panel ("Adaptive Brain" Card)

Displays: strategy weight bars (color-coded green/red/gray), per-strategy scorecard (win rate, avg PnL, trade count, streak), current regime with confidence %, threshold multipliers, total learning trades, last saved timestamp, learning on/off toggle.

**Verified:** 9/9 tests passed — weights adapt correctly (5x difference between winning/losing strategies), thresholds adjust on streaks, regime adjustments work, brain serializes under 5KB.

---

## 9. Risk Management Framework

Risk management is more important than strategy selection. A bad strategy with good risk management survives; a good strategy with bad risk management fails.

### Position Sizing Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Risk per trade | 2% of account max | Survive 5+ consecutive losses |
| Max single position | 5% of account | No single bet can ruin you |
| Max sector exposure | 20% of account | Diversification |
| Cash reserve | 20% minimum | Dry powder for opportunities |
| Max concurrent trades | 5 (normal), 11 (crypto mode) | Avoid overexposure |
| Min risk:reward | 1.5:1 | Only take asymmetric bets |

### Stop-Loss Methods

**Percentage-based:** Fixed % below entry (2–3% for daily, 0.8%–1.8% for tick-level scalping).

**ATR-based (recommended):** Stop = entry − (2 × ATR). Adapts to volatility automatically. In high volatility: multiply ATR by 1.3 (wider). In low volatility: multiply by 0.8 (tighter).

**Technical:** Stop below recent support level or below key moving average.

### Drawdown Management

| Drawdown Level | Action |
|---------------|--------|
| 5% | Warning — log it, show alert |
| 7.5% | Quant engine progressively de-risks |
| 10% | Reduce position sizes by 50% |
| 15% | Quant engine zeroes all positions. Full stop. |
| 20% | Review strategy before resuming |

Recovery math: a 50% loss requires a 100% gain to break even. A 30% loss requires 43%.

### Advanced Risk Metrics

**VaR (Value at Risk):** Maximum loss expected with 95% confidence over a time horizon. The quant engine calculates 1-day 99% VaR. Methods: historical (simplest, doesn't assume distribution), parametric (assumes normal — underestimates extremes), Monte Carlo (most flexible, slowest).

**CVaR / Expected Shortfall:** Average loss when losses *exceed* VaR. More useful than VaR because it captures tail risk. Two portfolios can have the same VaR but wildly different CVaR — the one with lower CVaR is safer.

**Factor Exposure (Fama-French):** Decompose returns into market risk, size factor, and value factor. Alpha = return after subtracting all factor contributions. If alpha is positive and significant, your stock picking adds value beyond just factor exposure.

---

## 10. Regime Detection

The single biggest improvement for any trading system is knowing what kind of market you're in and adjusting accordingly. No strategy works in all conditions.

### Four Market Regimes

**Regime 1 — Bull Trend:** Market trending up, VIX <15, ADX >25, RSI >50, MAs aligned (short > medium > long). Strategy: momentum/trend following. Avoid mean reversion.

**Regime 2 — Bear Trend:** Market trending down, VIX 15–25, ADX >25, RSI <50, MAs reversed. Strategy: trend following (short), be cautious buying dips. Avoid early dip-buying.

**Regime 3 — High Volatility Crash:** Sharp decline, VIX >30, extreme RSI, volume spiking, ATR >1.5x average. Strategy: mean reversion with tiny positions. Reduce all sizes by 50%+. Only high-confidence trades.

**Regime 4 — Choppy / Range-Bound:** Sideways, VIX <15 but no trend, ADX <20, RSI oscillating 30–70, MAs tangled. Strategy: mean reversion, sell resistance, buy support. Avoid trend following (constant whipsaws).

### Current Implementation

**Paper Trader (v5.0):** Measures directional consistency and return volatility across all assets every tick. Classifies as TRENDING (>55% consistency), CHOPPY (high vol + low consistency), or MIXED. Adjusts strategy weights accordingly.

**Quant Engine:** Uses volatility targeting to scale exposure across regimes. Drawdown protection progressively de-risks. The Momentum with Vol Break model specifically reduces exposure during volatility spikes.

### Research-Backed Enhancements (Planned)

**Hidden Markov Model (HMM):** Proven to be the best method for detecting regime shifts. Models markets as switching between hidden states (calm, volatile) based on observable returns/volatility. Multiple academic papers validate HMM outperforms other methods on S&P 500 data.

**Simplified GARCH(1,1) for browser:** Today's volatility = base + (weight × last return²) + (persistence × yesterday's volatility). Reacts faster to sudden volatility spikes than rolling standard deviation. Formula: `h_t = 0.00001 + 0.1 × return² + 0.85 × h_(t-1)`. Can be implemented in pure JavaScript with no libraries.

---

## 11. Position Sizing & Kelly Criterion

### Kelly Criterion

Mathematical formula for optimal bet size to maximize long-term growth. Developed in 1956, used by Warren Buffett and Bill Gross.

**Formula:** `Kelly % = W − (1 − W) / R`

Where W = win rate, R = avg win / avg loss.

**Example:** Win rate 55%, avg win $150, avg loss $100 (R = 1.5). Kelly % = 0.55 − (0.45 / 1.5) = 0.25 = 25%.

**Critical: Use fractional Kelly.**
- Full Kelly: mathematically optimal but produces huge drawdowns
- **Half-Kelly (recommended):** Multiply by 0.5. Captures ~75% of growth with much less volatility
- Quarter-Kelly: Very conservative, ~50% of growth, very stable

**Implementation rules:** Require minimum 30 trades before trusting Kelly numbers. Recalculate every 10 closed trades. Cap maximum position at 5% regardless of Kelly output. Calculate per-strategy — each signal type gets its own Kelly fraction.

### Volatility-Based Sizing

Position size = (account × risk %) / (ATR × price). Automatically gives larger positions in stable assets, smaller in volatile ones. In high-volatility regimes, reduce further. In low-volatility regimes, can increase slightly.

---

## 12. Backtesting Best Practices

### Critical Rules

**Out-of-sample testing:** Split data 70% training / 30% testing. Optimize only on training. Report results on test set only. If small parameter changes break the strategy, it's overfitted.

**No look-ahead bias:** Only use data available at time of decision. Never use tomorrow's close to predict tomorrow's entry.

**Realistic costs:** Account for commission (~0.1%), slippage (~0.2%), bid-ask spread. The quant engine models commissions (1 bps), spreads (2 bps), and market impact (Almgren-Chriss).

**Test across regimes:** Must work in bull markets, bear markets, sideways markets, and high-volatility crashes. If it only works in one condition, it's fragile.

### Validation Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Win Rate | ≥50% | <50% | — |
| Profit Factor | ≥1.5 | 1.2–1.5 | <1.2 |
| Sharpe Ratio | ≥1.0 | 0.5–1.0 | <0.5 |
| Max Drawdown | ≤20% | 20–30% | >30% |
| Sortino Ratio | ≥1.0 | — | — |
| Calmar Ratio | ≥1.0 | — | — |
| Recovery Factor | ≥2.0 | 1.0–2.0 | <1.0 |

### Common Mistakes

**Overfitting:** Fitting noise instead of signal. "80% win rate in backtest" usually means you memorized history. Solution: out-of-sample testing, robust parameters.

**Survivorship bias:** Only testing on stocks that survived (ignoring bankruptcies). Solution: use data that includes delisted stocks.

**Ignoring costs:** "25% annual return" becomes 15% after costs. Solution: always model transaction costs.

---

## 13. Performance Metrics Glossary

**Sharpe Ratio:** (average return − risk-free rate) / std dev of returns. Return per unit of total risk. >1.0 good, >2.0 excellent. The single most important number.

**Sortino Ratio:** Like Sharpe but only penalizes downside volatility. Better for scalping because big wins shouldn't count against you. >1.0 good, >2.0 excellent.

**Calmar Ratio:** Annualized return / maximum drawdown. Measures return relative to worst-case pain. >1.0 good, >2.0 excellent. Uses fewer data points than Sharpe/Sortino.

**Maximum Drawdown:** Largest peak-to-trough decline. The "worst day" number. Target <20%.

**Profit Factor:** Gross profits / gross losses. >1.0 = profitable, >1.5 = strong, >2.0 = excellent.

**Win Rate:** Percentage of profitable trades. For scalping, even 51% can be profitable with good risk/reward. Combine with profit factor for full picture.

**Recovery Factor:** Total return / max drawdown. How many times the strategy earns back its worst loss. >2.0 = resilient.

**Expectancy per Trade:** (win rate × avg win) − (loss rate × avg loss). Must be positive for the strategy to be worth running.

**VaR (Value at Risk):** Max expected loss at 95%/99% confidence. "95% of days, I lose less than $X."

**CVaR (Expected Shortfall):** Average loss on the worst 5% of days. Captures tail risk that VaR misses.

**HHI (Herfindahl-Hirschman Index):** Portfolio concentration. Lower = more diversified. 0.04 = perfectly spread across 25 assets.

---

## 14. Planned Enhancements

### Phase 1: Expand Adaptive Learning (Next Build)

The combined prompt for Claude Code to implement:

> Build all 4 adaptive learning features AND a full data persistence system so the bot genuinely gets smarter over time and never loses its progress between sessions.
>
> **1. Strategy Performance Tracking** — Track win rate and average PnL for each signal type (RSI, Momentum, MeanRev, MA Crossover, VWAP, Composite) as trades close. Keep a running scoreboard of every strategy's results.
>
> **2. Adaptive Signal Weighting** — Boost signals from profitable strategies, suppress losing ones. Start with conservative/slow adaptation rates — don't swing weights dramatically based on small samples.
>
> **3. Dynamic Threshold Adjustment** — On losing streaks: tighten stops, raise entry bar. On winning streaks: loosen slightly.
>
> **4. Regime Detection** — Track volatility and shift between trend-following in trending markets and mean-reversion in choppy markets.
>
> **5. Data Persistence** — Use localStorage to auto-save everything. Export/Import Brain buttons. Reset Learning button.
>
> **6. Visible UI Panel** — Live scorecard, strategy weights, regime indicator, trade count maturity, last saved timestamp.
>
> **7. Controls** — Toggle adaptive learning on/off. Export/Import/Reset Brain.

**Status:** ✅ Implemented in v5.0. All features verified with 9/9 tests passing.

### Phase 2: TensorFlow.js Neural Network

Add a browser-based neural network using TensorFlow.js (CDN, no backend) for price direction prediction:

- Small feedforward network: 10–15 features → 32 neurons → 16 neurons → 3 outputs (up/down/flat)
- Online learning: retrain every 20 closed trades on sliding window of 200–500 data points
- Model persistence: save/load via `model.save('localstorage://trading-model')`
- Confidence threshold: only act on predictions where confidence >60%
- Walk-forward validation to detect overfitting
- Anti-overfitting safeguards: minimum 50 trades before trusting, L2 regularization, rolling performance check

### Phase 3: Advanced Algorithms

- **LSTM layer** for time-series pattern recognition (30–50 tick sliding window)
- **GARCH-like volatility estimation** for better regime detection
- **Kelly Criterion position sizing** (half-Kelly, per-strategy)
- **ATR-based trailing stops** with regime-adjusted multipliers
- **Multi-timeframe confirmation** (1-min, 5-min, 15-min alignment)
- **Bayesian threshold optimization** (learn optimal RSI/momentum levels from data)
- **Feature importance tracking** (which indicators are actually predictive right now)
- **Ensemble method** (combine NN + rules + adaptive weights; disagreement = don't trade)

### Phase 4: Risk Dashboard

- Full drawdown circuit breaker (tiered: warning → reduce → pause → halt)
- Sharpe, Sortino, Calmar, Profit Factor — all calculated on rolling basis
- Per-strategy expectancy and Kelly fraction display
- Strategy correlation matrix (avoid strategies that all fail simultaneously)
- Performance attribution (what % of profits from which strategy/model)
- A/B testing toggle (ML on vs. off comparison)

### Libraries for Browser Implementation

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/simple-statistics@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

---

## 15. Configuration Reference

### Quant Engine (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tickers` | 28 diversified assets | Equity indices, bonds, commodities, mega-cap tech, financials, energy, crypto |
| `years` | 5 | Historical data lookback |
| `method` | risk_parity | Portfolio optimization (also: mean_variance, black_litterman) |
| `ensemble_method` | inverse_vol | Signal combination (also: equal_weight, performance_weighted) |
| `vol_target` | 10% | Annualized volatility target |
| `max_drawdown` | 15% | Hard stop — zeroes all positions |
| `max_position_size` | 15% | Maximum weight in any single asset |
| `rebalance_frequency` | 5 days | How often portfolio rebalances |

### Paper Trader (in-HTML config)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting capital | $1,000 | Simulated |
| Refresh interval | 6 seconds (default), 3s option | Tick frequency |
| Crypto assets | 14 | BTC through LTC |
| Risk levels | Moderate / Aggressive / YOLO | Affect TP/SL/trailing thresholds |
| Learning alpha | 0.15 | EMA decay for performance tracking |
| Min trades for weight adjustment | 5 | Prevents early noise |
| Weight range | 0.2x – 3.0x | Floor and ceiling for strategy weights |
| Brain localStorage key | `czwm_brain_v1` | ~1.7KB serialized |
| Regime confidence threshold | 40% | Below this, no regime adjustments applied |

---

## 16. Change Log

### v5.0 — Adaptive Learning Brain (2026-03-10)
Full ML learning system: strategy performance tracking, adaptive signal weighting, dynamic threshold adjustment, regime detection, localStorage persistence, Export/Import/Reset Brain, visible UI panel, learning toggle. 9/9 tests passed.

### v4.5 — Fix TP/SL Thresholds for Tick-Level Trading (2026-03-10)
Recalibrated all exit thresholds for tick-level data. TP: 1.2%–3.0%, SL: -0.8% to -1.8%, trailing: 0.5%–1.0%. Trailing activation lowered to 0.1%. All 3 exit types verified working.

### v4.4 — Fix Crypto Price Source (2026-03-10)
Replaced geo-blocked `api.binance.com` with `api.binance.us` + CoinGecko fallback. Root cause of zero trades. Verified: 15 ticks fetched, 3 trades executed within 12 seconds.

### v4.3 — Signal Tuning for Live Tick Data (2026-03-10)
Shortened indicator periods (MA 3/8, RSI 8, momentum lookback 5, z-score 8, VWAP 6). Lowered all signal thresholds. Added MicroTrend signal. Faster refresh (6s default, 3s option). Trades fire within 30–60 seconds.

### v4.2 — 24/7 Crypto Trading Mode (2026-03-10)
Expanded crypto universe to 14 assets. Auto crypto-only mode when market closed. Crypto signal boost 1.4x, +6 max positions, 1.3x sizing. `[24/7]` trade tagging.

### v4.1 — Quant Engine Dashboard & Bug Fixes (2026-03-10)
Fixed ML Alpha hang (scaled complexity with universe size). Fixed `plt.show()` blocking. New Flask server, web dashboard, requirements.txt. Research documents integrated.

---

## Research Gap Analysis

### What the existing research covers well:
- Core strategies (trend following, mean reversion, momentum, stat arb, factor investing) — thorough
- Technical indicators — comprehensive reference with practical combinations
- Risk management — strong coverage of Sharpe, Sortino, Kelly, position sizing, drawdown
- ML architectures — good overview of LSTM, Transformers, CNN, RL, NLP
- Backtesting — solid best practices and common pitfalls
- Advanced strategies — VaR, CVaR, factor models, pairs trading with cointegration

### What my deep research adds:

**Browser-specific ML implementation:** TensorFlow.js CDN integration, model save/load to localStorage, online learning patterns, ELM (Extreme Learning Machine) as a lightweight alternative for real-time in-browser training.

**Scalping-specific algorithms:** Tick-level feature engineering (15 specific features normalized for ML input), multi-timeframe confirmation (1/5/15 min alignment), microstructure features (bid-ask spread, order flow imbalance).

**GARCH volatility estimation:** Simplified browser-implementable version for more responsive volatility tracking than rolling standard deviation. Reacts faster to spikes, decays slowly — mirrors real market behavior.

**Hidden Markov Model justification:** Academic backing showing HMM outperforms other regime detection methods. Simplified two-state browser implementation without heavy libraries.

**Kelly Criterion implementation details:** Half-Kelly and quarter-Kelly recommendations, per-strategy calculation, minimum sample sizes (30 trades), recalculation frequency (every 10 trades), hard cap at 5%.

**Anti-overfitting safeguards for live learning:** Walk-forward validation, confidence thresholds (>60%), minimum sample sizes (50 trades before trusting ML), ensemble disagreement = don't trade, rolling performance checks with automatic fallback to rules.

**Drawdown circuit breaker:** Tiered automated response (warning → reduce → pause → halt) that the current system's linear drawdown protection doesn't cover.

**Reward-adjusted learning:** Evaluating strategies by Sharpe ratio rather than just win/loss. A 50% win rate with 3:1 reward/risk is better than 70% with 0.5:1.

---

## Key Principles

1. **Risk management > strategy selection.** Always. Non-negotiable.
2. **No single strategy works in all conditions.** Regime awareness is critical.
3. **The model must earn trust.** Minimum sample sizes before letting ML override rules.
4. **Persist everything.** Every trade, weight, model state — save to localStorage + JSON export.
5. **Display everything.** The UI should show what the bot is thinking. Transparency enables debugging.
6. **Start conservative, loosen gradually.** Better to miss good trades than blow up on bad ones.
7. **Ensemble everything.** Disagreement between models = don't trade.
8. **Backtest skeptically.** Out-of-sample testing, realistic costs, multiple market regimes.
9. **Simplicity over complexity.** Add layers only if they measurably improve results.
10. **Consistency > optimization.** A steady 12% beats a volatile 20%.
