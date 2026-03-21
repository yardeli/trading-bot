"""
Unified Trading Dashboard — Backtest + Live Trading in one UI.

Modes:
    Backtest: runs quant engine on historical data, streams equity curve
    Live:     connects to Alpaca paper trading, executes real orders

Usage:
    python dashboard.py                  # Start on port 8888
    python dashboard.py --port 9000
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# Quant engine modules are bundled in this repo

from alpha.mean_reversion import OUMeanReversion, PairsTrading
from alpha.ml_alpha import MLAlpha
from alpha.momentum import (
    CrossSectionalMomentum,
    MomentumWithVolBreak,
    TimeSeriesMomentum,
)
from backtest.engine import BacktestEngine
from config import SystemConfig
from data.feed import DataFeed
from ensemble.aggregator import SignalAggregator
from features.engine import FeatureEngine
from portfolio.optimizer import PortfolioOptimizer
from risk.manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Dashboard")

app = Flask(__name__)
CORS(app)

# ── JSON encoder ──────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)[:10]
        return super().default(obj)

import flask.json.provider as _fjp
_orig = _fjp.DefaultJSONProvider.dumps
def _patched(self, obj, **kw):
    kw.setdefault('cls', NumpyEncoder)
    return json.dumps(obj, **kw)
_fjp.DefaultJSONProvider.dumps = _patched

# ── Alpaca credentials ────────────────────────────────────────
ALPACA_KEY = os.environ.get("ALPACA_KEY", "PKA4ZJAUHC2XSXBMVFISRR7H2Q")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "FnbWgaFKFzqEkEoroSA9ioXPJUTZm1SGyM894pUfH1dE")

# ── Shared state ──────────────────────────────────────────────
state = {
    "mode": "backtest",       # backtest | live
    "status": "idle",         # idle | loading | running | complete | error | live_active | live_stopped
    "progress": 0,
    "progress_msg": "",
    "start_time": None,
    "elapsed": 0,
    "error": None,
    "equity_history": [],
    "current_day": 0,
    "total_days": 0,
    "initial_capital": 1_000_000,
    "trade_log": [],
    "current_weights": {},
    "model_weights": {},
    "risk_metrics": {},
    "current_signals": {},
    "metrics": {
        "total_return": 0, "ann_return": 0, "ann_vol": 0,
        "sharpe": 0, "sortino": 0, "max_dd": 0, "current_dd": 0,
        "win_rate": 0, "profit_factor": 0, "calmar": 0,
    },
    "final": None,
    # Live-specific
    "live_running": False,
    "live_interval": 600,
    "live_account": {},
    "live_positions": [],
    "last_rebalance": None,
    "next_rebalance": None,
    "market_open": False,
    "rebalance_count": 0,
}
lock = threading.Lock()
live_stop_event = threading.Event()


# ═══════════════════════════════════════════════════════════════
#  BACKTEST MODE
# ═══════════════════════════════════════════════════════════════

def _build_models(config):
    alpha_cfg = config.alpha
    models = []
    for name in alpha_cfg.enabled_models:
        if name == "ts_momentum": models.append(TimeSeriesMomentum(alpha_cfg))
        elif name == "xs_momentum": models.append(CrossSectionalMomentum(alpha_cfg))
        elif name == "momentum_vol_break": models.append(MomentumWithVolBreak(alpha_cfg))
        elif name == "ou_mean_reversion": models.append(OUMeanReversion(alpha_cfg))
        elif name == "pairs_trading": models.append(PairsTrading(alpha_cfg))
        elif name == "ml_alpha": models.append(MLAlpha(alpha_cfg))
    return models or [
        TimeSeriesMomentum(config.alpha), CrossSectionalMomentum(config.alpha),
        MomentumWithVolBreak(config.alpha), OUMeanReversion(config.alpha),
        PairsTrading(config.alpha), MLAlpha(config.alpha),
    ]


class DashboardCallback:
    def __init__(self):
        self.peak_equity = 0
        self.initial_capital = 1_000_000
        self.return_history = []  # only NON-ZERO returns (skip warmup)
        self.trading_started = False
        self.trading_start_equity = 0
        self.trading_days = 0

    def on_backtest_start(self, total_days, config_info, initial_capital):
        with lock:
            state["total_days"] = total_days
            state["current_day"] = 0
            state["equity_history"] = []
            state["trade_log"] = []
            state["status"] = "running"
            state["progress_msg"] = "Warming up models..."
            state["initial_capital"] = initial_capital
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.trading_started = False
        self.trading_start_equity = initial_capital
        self.trading_days = 0
        self.return_history = []

    def on_day_update(self, date, equity, weights=None, risk_metrics=None,
                      model_weights=None, trade_info=None):
        with lock:
            state["current_day"] += 1
            state["progress"] = round(state["current_day"] / max(state["total_days"], 1) * 100, 1)

            # Detect when trading actually starts (equity moves from initial capital)
            equity_changed = abs(equity - self.initial_capital) > 1.0
            if not self.trading_started and equity_changed:
                self.trading_started = True
                self.trading_start_equity = self.initial_capital
                self.peak_equity = self.initial_capital
                state["progress_msg"] = "Trading..."

            # Only add to equity history once trading starts (skip flat warmup)
            if self.trading_started:
                state["equity_history"].append({"date": str(date)[:10], "equity": round(equity, 2)})
                self.trading_days += 1
            else:
                # During warmup, just update progress message
                state["progress_msg"] = f"Warming up models... (day {state['current_day']})"
                return  # skip everything else during warmup

            if weights is not None:
                state["current_weights"] = {k: round(v, 4) for k, v in weights.items() if abs(v) > 0.001}
            if risk_metrics is not None:
                state["risk_metrics"] = {k: round(v, 6) if isinstance(v, float) else v for k, v in risk_metrics.items()}
            if model_weights is not None:
                state["model_weights"] = {k: round(v, 4) if v == v else 0 for k, v in model_weights.items()}
            if trade_info is not None:
                state["trade_log"].append({k: (round(v, 4) if isinstance(v, float) else str(v)[:10] if hasattr(v, 'strftime') else v) for k, v in trade_info.items()})

            # Compute returns only from actual trading days (no warmup zeros)
            eq_list = state["equity_history"]
            n = len(eq_list)
            if n > 1:
                prev = eq_list[-2]["equity"]
                ret = (equity - prev) / prev if prev else 0
                self.return_history.append(ret)

            self.peak_equity = max(self.peak_equity, equity)
            total_ret = equity / self.trading_start_equity - 1
            dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0

            # Use actual trading days for annualization (not warmup days)
            n_years = self.trading_days / 252
            ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0.05 else 0
            ann_vol = sharpe = sortino = win_rate = profit_factor = 0
            if len(self.return_history) > 10:
                rets = np.array(self.return_history)
                ann_vol = float(np.std(rets) * np.sqrt(252))
                if ann_vol > 0: sharpe = ann_ret / ann_vol
                down = rets[rets < 0]
                dv = float(np.std(down) * np.sqrt(252)) if len(down) > 0 else ann_vol
                sortino = ann_ret / dv if dv > 0 else 0
                win_rate = float(np.mean(rets > 0))
                g, l = float(np.sum(rets[rets > 0])), float(np.abs(np.sum(rets[rets < 0])))
                profit_factor = g / l if l > 0 else 0
            prev_dd = state["metrics"].get("max_dd", 0)
            state["metrics"] = {
                "total_return": round(total_ret, 6), "ann_return": round(ann_ret, 6),
                "ann_vol": round(ann_vol, 6), "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4), "max_dd": round(min(dd, prev_dd), 6),
                "current_dd": round(dd, 6), "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 4),
                "calmar": round(ann_ret / abs(min(dd, prev_dd)) if min(dd, prev_dd) < -0.001 else 0, 4),
            }

    def on_backtest_complete(self):
        with lock:
            state["progress"] = 100
            state["progress_msg"] = "Complete"

    def show_final_report(self, result):
        pass


def _run_backtest(config_overrides):
    try:
        with lock:
            state.update({"status": "loading", "progress": 0, "progress_msg": "Initializing...",
                          "error": None, "final": None, "equity_history": [], "trade_log": [],
                          "start_time": time.time(), "mode": "backtest"})
        config = SystemConfig()
        if config_overrides.get("tickers"):
            config.data.tickers = config_overrides["tickers"]
        if "years" in config_overrides:
            config.data.years = int(config_overrides["years"])
        capital = float(config_overrides.get("capital", 1_000_000))

        with lock: state.update({"progress_msg": "Downloading market data...", "progress": 5})
        feed = DataFeed(config.data)
        feed.fetch()
        logger.info(f"Data loaded: {feed.prices.shape}")

        with lock: state.update({"progress_msg": "Building alpha models...", "progress": 15})
        models = _build_models(config)

        with lock: state.update({"progress_msg": "Running backtest...", "progress": 20})
        cb = DashboardCallback()
        cb.on_backtest_start(len(feed.prices), {}, capital)
        engine = BacktestEngine(config)
        result = engine.run(feed, models, initial_capital=capital, dashboard=cb)

        perf, ex = result.performance_metrics, result.execution_summary
        detailed = [{"date": str(t.get("date", ""))[:10], "n_trades": t.get("n_trades", 0),
                      "turnover": round(t.get("turnover", 0), 4), "cost": round(t.get("total_cost", 0), 2),
                      "port_value": round(t.get("port_value", 0), 2)} for t in result.trade_log]
        eq_dates = [str(d)[:10] for d in result.equity_curve.index]
        eq_vals = [round(v, 2) for v in result.equity_curve.values]
        cumret = result.equity_curve / result.equity_curve.iloc[0]
        drawdown = ((cumret - cumret.cummax()) / cumret.cummax()).values.tolist()

        with lock:
            state.update({"status": "complete", "progress": 100, "progress_msg": "Complete",
                          "elapsed": round(time.time() - state["start_time"], 1), "trade_log": detailed,
                          "final": {
                              "performance": {k: round(v, 6) if isinstance(v, float) else v for k, v in perf.items()},
                              "execution": {k: round(v, 4) if isinstance(v, float) else v for k, v in ex.items()},
                              "equity_curve": {"dates": eq_dates, "values": eq_vals},
                              "drawdown": [round(v, 6) for v in drawdown],
                              "detailed_trades": detailed}})
        logger.info(f"Backtest complete in {state['elapsed']}s")
    except Exception as e:
        logger.exception("Backtest failed")
        with lock: state.update({"status": "error", "error": str(e), "progress_msg": f"Error: {e}"})


# ═══════════════════════════════════════════════════════════════
#  LIVE TRADING MODE
# ═══════════════════════════════════════════════════════════════

def _get_alpaca_clients():
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    tc = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
    dc = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    return tc, dc


def _update_account_positions(tc):
    """Refresh account + positions in state."""
    try:
        acct = tc.get_account()
        positions = tc.get_all_positions()
        clock = tc.get_clock()
        with lock:
            state["live_account"] = {
                "equity": round(float(acct.equity), 2),
                "cash": round(float(acct.cash), 2),
                "buying_power": round(float(acct.buying_power), 2),
                "portfolio_value": round(float(acct.portfolio_value), 2),
                "long_market_value": round(float(acct.long_market_value), 2),
                "short_market_value": round(float(acct.short_market_value), 2),
            }
            state["live_positions"] = [{
                "symbol": p.symbol, "qty": float(p.qty),
                "market_value": round(float(p.market_value), 2),
                "unrealized_pl": round(float(p.unrealized_pl), 2),
                "unrealized_plpc": round(float(p.unrealized_plpc) * 100, 2),
                "current_price": round(float(p.current_price), 2),
                "avg_entry_price": round(float(p.avg_entry_price), 2),
                "side": "long" if float(p.qty) > 0 else "short",
            } for p in positions]
            state["market_open"] = clock.is_open
    except Exception as e:
        logger.error(f"Account update failed: {e}")


def _live_rebalance(tc, config):
    """Run one rebalance cycle: data -> signals -> optimize -> trade."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    logger.info("=" * 50 + " LIVE REBALANCE " + "=" * 50)
    with lock:
        state["progress_msg"] = "Fetching historical data..."

    # 1. Map Alpaca crypto symbols (BTC/USD) to yfinance (BTC-USD) for data fetch
    alpaca_to_yf = {}
    yf_tickers = []
    for t in config.data.tickers:
        if '/' in t:
            yf_sym = t.replace('/', '-')
            alpaca_to_yf[yf_sym] = t
            yf_tickers.append(yf_sym)
        else:
            alpaca_to_yf[t] = t
            yf_tickers.append(t)

    data_config_copy = SystemConfig().data
    data_config_copy.tickers = yf_tickers
    feed = DataFeed(data_config_copy)
    feed.load()

    # Rename columns back to Alpaca symbols for trading
    rename_map = {yf: alp for yf, alp in alpaca_to_yf.items() if yf != alp}
    if rename_map:
        feed.prices = feed.prices.rename(columns=rename_map)
        feed.returns = feed.returns.rename(columns=rename_map)
        if feed.volume is not None: feed.volume = feed.volume.rename(columns=rename_map)
        if feed.high is not None: feed.high = feed.high.rename(columns=rename_map)
        if feed.low is not None: feed.low = feed.low.rename(columns=rename_map)

    tradeable = list(feed.prices.columns)

    # 2. Features + signals
    with lock: state["progress_msg"] = "Generating signals..."
    feat_engine = FeatureEngine(config.features)
    features = feat_engine.generate(feed)
    models = _build_models(config)
    all_signals = {}
    for m in models:
        try:
            all_signals[m.name] = m.generate_signals(feed, features)
        except Exception as e:
            logger.warning(f"  {m.name} failed: {e}")

    # 3. Aggregate
    agg = SignalAggregator(method=config.alpha.ensemble_method, ic_lookback=63)
    returns = feed.returns
    N = 300
    recent_ret = returns.iloc[max(0, len(returns)-N):]
    date_signals = {n: s.iloc[max(0, len(s)-N):] for n, s in all_signals.items()}
    combined = agg.aggregate(date_signals, recent_ret)
    signal_row = combined.iloc[-1]

    with lock:
        state["current_signals"] = {k: round(float(v), 4) for k, v in signal_row.items() if abs(v) > 0.005}
        state["model_weights"] = {k: round(float(v), 4) for k, v in agg.model_weights.items()}

    # 4. Portfolio optimization
    with lock: state["progress_msg"] = "Optimizing portfolio..."
    opt = PortfolioOptimizer(config.portfolio)
    cov = PortfolioOptimizer.estimate_covariance(returns.iloc[-252:], method="exponential", halflife=63)

    acct = tc.get_account()
    equity = float(acct.equity)
    positions = tc.get_all_positions()
    cur_weights = pd.Series(0.0, index=tradeable)
    for p in positions:
        if p.symbol in cur_weights.index:
            cur_weights[p.symbol] = float(p.market_value) / equity

    target_weights = opt.optimize(signal_row, cov, cur_weights)

    # 5. Risk management
    rm = RiskManager(config.risk)
    eq_hist = [e["equity"] for e in state.get("equity_history", [])] or [equity]
    eq_series = pd.Series(eq_hist, index=pd.date_range(end=pd.Timestamp.now(), periods=len(eq_hist)))
    target_weights = rm.check_and_adjust(target_weights, returns.iloc[-252:], eq_series)

    with lock:
        state["current_weights"] = {k: round(float(v), 4) for k, v in target_weights.items() if abs(v) > 0.001}
        state["risk_metrics"] = rm.get_risk_report()

    # 6. Execute trades — carefully manage cash and avoid impossible orders
    with lock: state["progress_msg"] = "Executing trades..."
    trades_made = []

    # Build a map of what we currently hold
    held_symbols = {}
    for p in positions:
        held_symbols[p.symbol] = float(p.market_value)

    # Compute all desired trades
    desired_trades = []
    for ticker, tw in target_weights.items():
        target_value = equity * tw
        current_value = held_symbols.get(ticker, 0)
        trade_value = target_value - current_value

        if abs(trade_value) < equity * 0.005:
            continue  # skip tiny trades

        is_crypto = '/' in ticker

        # Skip short-sells on crypto (Alpaca doesn't allow it)
        if is_crypto and trade_value < 0 and current_value <= 0:
            continue

        # Close position if target weight is ~0
        if abs(tw) < 0.001 and current_value > 0:
            desired_trades.append({"ticker": ticker, "action": "close", "value": current_value, "crypto": is_crypto})
            continue

        if trade_value > 0:
            desired_trades.append({"ticker": ticker, "action": "buy", "value": abs(trade_value), "crypto": is_crypto})
        elif trade_value < 0 and current_value > 0:
            # Only sell what we actually hold
            sell_value = min(abs(trade_value), current_value)
            desired_trades.append({"ticker": ticker, "action": "sell", "value": sell_value, "crypto": is_crypto})

    # Execute SELLS and CLOSES first (frees up cash), then BUYS
    sells = [t for t in desired_trades if t["action"] in ("sell", "close")]
    buys = [t for t in desired_trades if t["action"] == "buy"]

    # Track available cash
    available_cash = float(tc.get_account().cash)

    for trade in sells:
        ticker = trade["ticker"]
        if trade["action"] == "close":
            try:
                tc.close_position(ticker)
                available_cash += trade["value"]
                trades_made.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": ticker,
                                     "side": "CLOSE", "notional": round(trade["value"], 2), "status": "filled"})
                logger.info(f"  CLOSE {ticker} (+${trade['value']:,.0f} cash)")
            except Exception as e:
                trades_made.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": ticker,
                                     "side": "CLOSE", "notional": 0, "status": f"error: {e}"})
        else:
            notional = round(trade["value"], 2)
            try:
                tif = TimeInForce.GTC if trade["crypto"] else TimeInForce.DAY
                order = tc.submit_order(MarketOrderRequest(
                    symbol=ticker, notional=notional, side=OrderSide.SELL, time_in_force=tif))
                available_cash += notional
                trades_made.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": ticker,
                                     "side": "sell", "notional": notional, "status": "submitted",
                                     "order_id": str(order.id)[:8]})
                logger.info(f"  SELL ${notional:,.0f} of {ticker}")
            except Exception as e:
                trades_made.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": ticker,
                                     "side": "sell", "notional": notional, "status": f"error: {e}"})

    # Small delay to let sells settle
    if sells:
        import time as _time
        _time.sleep(1)
        available_cash = float(tc.get_account().cash)

    # Now execute buys, respecting available cash
    for trade in buys:
        ticker = trade["ticker"]
        notional = round(min(trade["value"], available_cash * 0.95), 2)  # Keep 5% buffer
        if notional < 1:
            continue

        try:
            tif = TimeInForce.GTC if trade["crypto"] else TimeInForce.DAY
            order = tc.submit_order(MarketOrderRequest(
                symbol=ticker, notional=notional, side=OrderSide.BUY, time_in_force=tif))
            available_cash -= notional
            trades_made.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": ticker,
                                 "side": "buy", "notional": notional, "status": "submitted",
                                 "order_id": str(order.id)[:8]})
            logger.info(f"  BUY ${notional:,.0f} of {ticker} (cash left: ${available_cash:,.0f})")
        except Exception as e:
            trades_made.append({"time": datetime.now().strftime("%H:%M:%S"), "symbol": ticker,
                                 "side": "buy", "notional": notional, "status": f"error: {e}"})
            logger.warning(f"  Failed {ticker}: {e}")

    # 7. Record
    _update_account_positions(tc)
    eq_now = float(tc.get_account().equity)
    with lock:
        state["equity_history"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "equity": round(eq_now, 2)})
        state["trade_log"].extend(trades_made)
        state["last_rebalance"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["rebalance_count"] += 1
        state["progress_msg"] = f"Active — {len(trades_made)} trades executed"

    logger.info(f"Rebalance #{state['rebalance_count']} complete: {len(trades_made)} trades")


def _run_live_trading(interval):
    """Main live trading loop."""
    live_stop_event.clear()
    try:
        tc, dc = _get_alpaca_clients()
        acct = tc.get_account()
        logger.info(f"Alpaca connected. Equity: ${float(acct.equity):,.2f}")

        config = SystemConfig()

        # Check if market is open to decide which tickers to use
        clock = tc.get_clock()
        is_market_open = clock.is_open

        # Crypto tickers trade 24/7 — use these when stock market is closed
        CRYPTO_TICKERS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD",
                          "LINK/USD", "DOT/USD", "LTC/USD"]

        if is_market_open:
            # Use full universe: stocks + crypto
            candidate_tickers = config.data.tickers + CRYPTO_TICKERS
        else:
            # Weekend/after-hours: crypto only (always tradeable)
            candidate_tickers = CRYPTO_TICKERS
            logger.info("Market closed — switching to crypto-only mode (trades 24/7)")

        # Filter to tradeable
        tradeable = []
        for t in candidate_tickers:
            try:
                asset = tc.get_asset(t)
                if asset.tradable: tradeable.append(t)
            except: pass
        config.data.tickers = tradeable
        logger.info(f"Tradeable: {len(tradeable)} tickers ({', '.join(tradeable[:8])}{'...' if len(tradeable)>8 else ''})")

        init_equity = float(acct.equity)
        with lock:
            state.update({
                "mode": "live", "status": "live_active", "live_running": True,
                "start_time": time.time(), "error": None, "equity_history": [],
                "trade_log": [], "initial_capital": init_equity, "rebalance_count": 0,
                "progress": 100, "progress_msg": "Connecting to Alpaca...",
                "live_interval": interval,
                "metrics": {"total_return": 0, "ann_return": 0, "ann_vol": 0,
                            "sharpe": 0, "sortino": 0, "max_dd": 0, "current_dd": 0,
                            "win_rate": 0, "profit_factor": 0, "calmar": 0},
            })

        _update_account_positions(tc)

        # Initial rebalance
        try:
            _live_rebalance(tc, config)
        except Exception as e:
            logger.error(f"Initial rebalance failed: {e}")
            with lock: state["progress_msg"] = f"Rebalance failed: {e}"

        # Loop
        while not live_stop_event.is_set():
            with lock:
                state["next_rebalance"] = datetime.fromtimestamp(
                    time.time() + interval).strftime("%H:%M:%S")
                state["elapsed"] = round(time.time() - state["start_time"], 1)

            # Wait for interval, checking stop event every second
            for _ in range(interval):
                if live_stop_event.is_set():
                    break
                time.sleep(1)
                # Refresh account every 30s
                if _ % 30 == 0:
                    _update_account_positions(tc)
                    eq = float(tc.get_account().equity)
                    with lock:
                        state["elapsed"] = round(time.time() - state["start_time"], 1)
                        # Update running P&L
                        total_ret = (eq / init_equity) - 1
                        state["metrics"]["total_return"] = round(total_ret, 6)

            if live_stop_event.is_set():
                break

            # Rebalance — always runs (crypto trades 24/7)
            try:
                clock = tc.get_clock()
                with lock: state["market_open"] = clock.is_open
                _live_rebalance(tc, config)
            except Exception as e:
                logger.error(f"Rebalance error: {e}")
                with lock: state["progress_msg"] = f"Error: {e}"

        with lock:
            state["status"] = "live_stopped"
            state["live_running"] = False
            state["progress_msg"] = "Live trading stopped"
        logger.info("Live trading stopped")

    except Exception as e:
        logger.exception("Live trading failed")
        with lock:
            state.update({"status": "error", "error": str(e), "live_running": False,
                          "progress_msg": f"Error: {e}"})


# ═══════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    with lock:
        eq = state["equity_history"]
        if len(eq) > 1000:
            step = len(eq) // 1000
            eq = eq[::step]
        return jsonify({
            "mode": state["mode"],
            "status": state["status"],
            "progress": state["progress"],
            "progress_msg": state["progress_msg"],
            "error": state["error"],
            "elapsed": round(time.time() - state["start_time"], 1) if state["start_time"] else 0,
            "current_day": state["current_day"],
            "total_days": state["total_days"],
            "equity_history": eq,
            "current_weights": state["current_weights"],
            "model_weights": state["model_weights"],
            "risk_metrics": state["risk_metrics"],
            "current_signals": state["current_signals"],
            "metrics": state["metrics"],
            "trade_count": len(state["trade_log"]),
            # Live-specific
            "live_running": state["live_running"],
            "live_interval": state["live_interval"],
            "live_account": state["live_account"],
            "live_positions": state["live_positions"],
            "last_rebalance": state["last_rebalance"],
            "next_rebalance": state["next_rebalance"],
            "market_open": state["market_open"],
            "rebalance_count": state["rebalance_count"],
        })

@app.route("/api/trades")
def api_trades():
    with lock:
        page = int(request.args.get("page", 0))
        size = int(request.args.get("size", 50))
        trades = state["trade_log"]
        start = max(0, len(trades) - (page + 1) * size)
        end = len(trades) - page * size
        return jsonify({
            "trades": list(reversed(trades[start:end])),
            "total": len(trades),
            "page": page,
            "pages": max(1, (len(trades) + size - 1) // size),
            "mode": state["mode"],
        })

@app.route("/api/result")
def api_result():
    with lock:
        if state["final"] is None:
            return jsonify({"error": "No results yet"}), 404
        return jsonify(state["final"])

@app.route("/api/start", methods=["POST"])
def api_start():
    with lock:
        if state["status"] in ("loading", "running", "live_active"):
            return jsonify({"error": "Already running"}), 409
    body = request.json or {}
    mode = body.pop("mode", "backtest")
    if mode == "live":
        interval = int(body.get("interval", 600))
        t = threading.Thread(target=_run_live_trading, args=(interval,), daemon=True)
        t.start()
        return jsonify({"status": "live_started"})
    else:
        t = threading.Thread(target=_run_backtest, args=(body,), daemon=True)
        t.start()
        return jsonify({"status": "backtest_started"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    live_stop_event.set()
    return jsonify({"status": "stop_requested"})

@app.route("/api/force-rebalance", methods=["POST"])
def api_force_rebalance():
    with lock:
        if not state["live_running"]:
            return jsonify({"error": "Not in live mode"}), 400
    def _do():
        try:
            tc, _ = _get_alpaca_clients()
            config = SystemConfig()
            tradeable = []
            for t in config.data.tickers:
                try:
                    a = tc.get_asset(t)
                    if a.tradable: tradeable.append(t)
                except: pass
            config.data.tickers = tradeable
            _live_rebalance(tc, config)
        except Exception as e:
            logger.error(f"Force rebalance failed: {e}")
    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"status": "rebalance_triggered"})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    live_stop_event.set()
    time.sleep(0.5)
    with lock:
        state.update({
            "mode": "backtest", "status": "idle", "progress": 0, "progress_msg": "",
            "error": None, "final": None, "equity_history": [], "trade_log": [],
            "current_weights": {}, "model_weights": {}, "risk_metrics": {},
            "current_signals": {}, "live_running": False, "live_account": {},
            "live_positions": [], "last_rebalance": None, "next_rebalance": None,
            "rebalance_count": 0,
            "metrics": {"total_return": 0, "ann_return": 0, "ann_vol": 0,
                        "sharpe": 0, "sortino": 0, "max_dd": 0, "current_dd": 0,
                        "win_rate": 0, "profit_factor": 0, "calmar": 0},
        })
    return jsonify({"status": "reset"})

@app.route("/")
def index():
    return Response(DASHBOARD_HTML, mimetype="text/html")


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quant Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg0:#06090f;--bg1:#0c1220;--bg2:#131c2e;--bg3:#1a2540;--bg4:#213050;
  --border:#1e2d45;--border-light:#2a3f5f;
  --text:#e8edf5;--text2:#94a3b8;--text3:#5a6a84;
  --green:#10b981;--green-bg:rgba(16,185,129,.12);--green-dim:#065f46;
  --red:#f43f5e;--red-bg:rgba(244,63,94,.12);--red-dim:#881337;
  --blue:#3b82f6;--blue-bg:rgba(59,130,246,.12);
  --purple:#a78bfa;--purple-bg:rgba(167,139,250,.12);
  --cyan:#22d3ee;--cyan-bg:rgba(34,211,238,.12);
  --orange:#fb923c;--orange-bg:rgba(251,146,60,.12);
  --yellow:#facc15;--yellow-bg:rgba(250,204,21,.12);
  --gold:#d4a017;--gold-bg:rgba(212,160,23,.12);
}
body{font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;background:var(--bg0);color:var(--text);min-height:100vh}
.header{background:linear-gradient(180deg,#0f1728,var(--bg1));border-bottom:1px solid var(--border);padding:12px 24px;position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.brand{font-size:18px;font-weight:800;color:var(--gold);letter-spacing:-.5px}
.brand small{font-size:10px;color:var(--text3);font-weight:400;letter-spacing:1px;display:block;margin-top:-2px}
.hdr-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.badge-idle{background:var(--blue-bg);color:var(--blue)}
.badge-running{background:var(--green-bg);color:var(--green);animation:pulse 1.5s infinite}
.badge-complete{background:var(--purple-bg);color:var(--purple)}
.badge-error{background:var(--red-bg);color:var(--red)}
.badge-live{background:var(--orange-bg);color:var(--orange);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.btn{padding:7px 16px;border-radius:6px;border:1px solid var(--border);background:var(--bg3);color:var(--text);font-family:inherit;font-size:11px;cursor:pointer;transition:.15s;font-weight:700}
.btn:hover{background:var(--bg4);border-color:var(--border-light)}
.btn-start{background:var(--green-dim);border-color:var(--green);color:var(--green);font-size:13px;padding:9px 28px;letter-spacing:1px}
.btn-start:hover{background:#047857}
.btn-start:disabled{opacity:.3;cursor:not-allowed}
.btn-stop{background:var(--red-dim);border-color:var(--red);color:var(--red)}
.btn-stop:hover{background:#9f1239}
.btn-rebal{background:var(--bg3);border-color:var(--cyan);color:var(--cyan);font-size:10px}
.btn-reset{font-size:10px;color:var(--text3)}
.elapsed{font-size:11px;color:var(--text3)}
/* Mode toggle */
.mode-toggle{display:flex;border:1px solid var(--border);border-radius:6px;overflow:hidden}
.mode-btn{padding:6px 16px;font-size:11px;font-family:inherit;border:none;cursor:pointer;font-weight:700;transition:.15s;background:var(--bg3);color:var(--text3)}
.mode-btn.active{background:var(--gold);color:var(--bg0)}
.mode-btn:hover:not(.active){background:var(--bg4);color:var(--text2)}
/* Interval selector */
.interval-sel{background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:5px 8px;font-family:inherit;font-size:10px}
.main{max-width:1600px;margin:0 auto;padding:14px}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.hero{grid-column:1/-1}
.port-val{font-size:44px;font-weight:800;line-height:1}
.port-chg{font-size:14px;margin-top:4px}
.port-stats{display:flex;gap:20px;flex-wrap:wrap}
.stat{display:flex;flex-direction:column;gap:1px}
.stat-l{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px}
.stat-v{font-size:13px;font-weight:700}
.progress-wrap{grid-column:1/-1}
.prog-bar{height:18px;background:var(--bg1);border-radius:9px;overflow:hidden;border:1px solid var(--border)}
.prog-fill{height:100%;border-radius:9px;transition:width .4s}
.prog-fill-bt{background:linear-gradient(90deg,var(--green-dim),var(--green))}
.prog-fill-live{background:linear-gradient(90deg,#92400e,var(--orange))}
.prog-label{font-size:10px;color:var(--text3);margin-top:4px;text-align:center}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.card-h{padding:8px 14px;border-bottom:1px solid var(--border);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--text3);display:flex;align-items:center;justify-content:space-between}
.card-b{padding:14px}
.chart-area{grid-column:1/3;min-height:300px}
.chart-area .card-b{padding:8px;height:280px}
.chart-dd{grid-column:1/3;min-height:170px}
.chart-dd .card-b{padding:8px;height:150px}
.m-row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(30,45,69,.3);font-size:12px}
.m-row:last-child{border-bottom:none}
.m-label{color:var(--text2)}.m-val{font-weight:700}
.w-bar-wrap{margin:4px 0}
.w-bar-label{display:flex;justify-content:space-between;font-size:10px;margin-bottom:2px}
.w-bar-bg{height:12px;background:var(--bg0);border-radius:6px;overflow:hidden}
.w-bar-fill{height:100%;border-radius:6px;transition:width .4s}
.trades-card{grid-column:1/-1}
.trades-table{width:100%;border-collapse:collapse;font-size:11px}
.trades-table th{text-align:left;padding:6px 10px;color:var(--text3);border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0;background:var(--bg2)}
.trades-table td{padding:6px 10px;border-bottom:1px solid rgba(30,45,69,.3)}
.trades-table tr:hover td{background:var(--bg3)}
.trades-scroll{max-height:400px;overflow-y:auto}
.page-controls{display:flex;justify-content:center;gap:6px;padding:8px;font-size:11px}
.page-btn{padding:3px 10px;border-radius:4px;border:1px solid var(--border);background:var(--bg3);color:var(--text2);cursor:pointer;font-family:inherit;font-size:10px}
.page-btn:hover{background:var(--bg4)}
.page-btn.active{background:var(--gold);color:var(--bg0);border-color:var(--gold)}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:12px}
.tab{padding:8px 18px;font-size:11px;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;transition:.15s;background:none;border-top:none;border-left:none;border-right:none;font-family:inherit}
.tab:hover{color:var(--text2)}.tab.active{color:var(--gold);border-bottom-color:var(--gold)}
.tab-panel{display:none}.tab-panel.active{display:block}
.pos{color:var(--green)}.neg{color:var(--red)}
.empty{color:var(--text3);font-size:12px;text-align:center;padding:40px 0}
/* Live info bar */
.live-bar{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.live-card{background:var(--bg1);border:1px solid var(--border);border-radius:6px;padding:10px 12px}
.live-card .lc-l{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.live-card .lc-v{font-size:16px;font-weight:800}
/* Positions */
.pos-table{width:100%;border-collapse:collapse;font-size:11px}
.pos-table th{text-align:left;padding:5px 8px;color:var(--text3);border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase}
.pos-table td{padding:5px 8px;border-bottom:1px solid rgba(30,45,69,.3)}
.pos-table tr:hover td{background:var(--bg3)}
@media(max-width:900px){.grid{grid-template-columns:1fr}.chart-area,.chart-dd,.trades-card{grid-column:1/-1}.live-bar{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>

<div class="header">
  <div class="brand">Quant Trading Dashboard<small>Backtest & Live Paper Trading</small></div>
  <div class="hdr-right">
    <div class="mode-toggle">
      <button class="mode-btn active" id="mode-bt" onclick="setMode('backtest')">Backtest</button>
      <button class="mode-btn" id="mode-live" onclick="setMode('live')">Live Trading</button>
    </div>
    <select class="interval-sel" id="interval-sel" style="display:none" title="Rebalance interval">
      <option value="120">2 min</option>
      <option value="300">5 min</option>
      <option value="600" selected>10 min</option>
      <option value="1800">30 min</option>
      <option value="3600">1 hour</option>
    </select>
    <span id="status-badge" class="badge badge-idle">IDLE</span>
    <span id="elapsed" class="elapsed"></span>
    <button id="btn-start" class="btn btn-start" onclick="doStart()">START</button>
    <button id="btn-stop" class="btn btn-stop" style="display:none" onclick="doStop()">STOP</button>
    <button id="btn-rebal" class="btn btn-rebal" style="display:none" onclick="doRebalance()">REBALANCE NOW</button>
    <button class="btn btn-reset" onclick="doReset()">RESET</button>
  </div>
</div>

<div class="main">
<div class="tabs">
  <button class="tab active" onclick="switchTab('dashboard',this)">Dashboard</button>
  <button class="tab" onclick="switchTab('trades',this)">Trade Log</button>
  <button class="tab" onclick="switchTab('positions',this)" id="tab-pos-btn">Positions</button>
  <button class="tab" onclick="switchTab('weights',this)">Weights & Risk</button>
</div>

<!-- Dashboard Tab -->
<div id="tab-dashboard" class="tab-panel active">
<div class="grid">

  <!-- Live account bar (hidden in backtest mode) -->
  <div class="live-bar" id="live-bar" style="display:none">
    <div class="live-card"><div class="lc-l">Account Equity</div><div class="lc-v" id="la-equity">—</div></div>
    <div class="live-card"><div class="lc-l">Cash</div><div class="lc-v" id="la-cash">—</div></div>
    <div class="live-card"><div class="lc-l">Buying Power</div><div class="lc-v" id="la-bp">—</div></div>
    <div class="live-card"><div class="lc-l">Market</div><div class="lc-v" id="la-market">—</div></div>
    <div class="live-card"><div class="lc-l">Last Rebalance</div><div class="lc-v" id="la-last-rb" style="font-size:11px">—</div></div>
    <div class="live-card"><div class="lc-l">Next Rebalance</div><div class="lc-v" id="la-next-rb" style="font-size:11px">—</div></div>
    <div class="live-card"><div class="lc-l">Rebalances</div><div class="lc-v" id="la-rb-count">0</div></div>
    <div class="live-card"><div class="lc-l">Open Positions</div><div class="lc-v" id="la-pos-count">0</div></div>
  </div>

  <!-- Hero -->
  <div class="hero card">
    <div class="card-b" style="width:100%">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px">
        <div>
          <div class="port-val" id="port-val">$1,000,000</div>
          <div class="port-chg" id="port-chg">+0.00%</div>
        </div>
        <div class="port-stats">
          <div class="stat"><span class="stat-l">Ann. Return</span><span class="stat-v" id="s-ann-ret">—</span></div>
          <div class="stat"><span class="stat-l">Sharpe</span><span class="stat-v" id="s-sharpe">—</span></div>
          <div class="stat"><span class="stat-l">Sortino</span><span class="stat-v" id="s-sortino">—</span></div>
          <div class="stat"><span class="stat-l">Max DD</span><span class="stat-v" id="s-max-dd">—</span></div>
          <div class="stat"><span class="stat-l">Win Rate</span><span class="stat-v" id="s-win-rate">—</span></div>
          <div class="stat"><span class="stat-l">Profit Factor</span><span class="stat-v" id="s-pf">—</span></div>
          <div class="stat"><span class="stat-l">Calmar</span><span class="stat-v" id="s-calmar">—</span></div>
          <div class="stat"><span class="stat-l">Trades</span><span class="stat-v" id="s-trades">0</span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="progress-wrap" id="progress-wrap">
    <div class="prog-bar"><div class="prog-fill prog-fill-bt" id="prog-fill" style="width:0%"></div></div>
    <div class="prog-label" id="prog-label">Press START to begin</div>
  </div>

  <div class="chart-area card">
    <div class="card-h"><span id="chart-title">Equity Curve</span><span id="chart-info"></span></div>
    <div class="card-b"><canvas id="equityChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-h">Performance</div>
    <div class="card-b" id="metrics-body"><div class="empty">Run a backtest or start live trading</div></div>
  </div>
  <div class="chart-dd card">
    <div class="card-h">Drawdown</div>
    <div class="card-b"><canvas id="ddChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-h">Model Weights</div>
    <div class="card-b" id="model-weights-body"><div class="empty">Waiting for data...</div></div>
  </div>
</div>
</div>

<!-- Trade Log Tab -->
<div id="tab-trades" class="tab-panel">
<div class="card trades-card">
  <div class="card-h"><span>Trade Log</span><span id="trade-count">0 trades</span></div>
  <div class="trades-scroll" id="trades-scroll">
    <table class="trades-table" id="trades-table">
      <thead id="trades-thead"></thead>
      <tbody id="trades-body"><tr><td colspan="6" class="empty">No trades yet</td></tr></tbody>
    </table>
  </div>
  <div class="page-controls" id="page-controls"></div>
</div>
</div>

<!-- Positions Tab -->
<div id="tab-positions" class="tab-panel">
<div class="card" style="grid-column:1/-1">
  <div class="card-h"><span>Open Positions</span><span id="pos-count">0 positions</span></div>
  <div class="card-b" id="positions-body"><div class="empty">No positions</div></div>
</div>
</div>

<!-- Weights & Risk Tab -->
<div id="tab-weights" class="tab-panel">
<div class="grid">
  <div class="card" style="grid-column:1/3">
    <div class="card-h">Current Portfolio Weights</div>
    <div class="card-b" id="weights-body"><div class="empty">No positions</div></div>
  </div>
  <div class="card">
    <div class="card-h">Risk Metrics</div>
    <div class="card-b" id="risk-body"><div class="empty">No data</div></div>
  </div>
</div>
</div>

</div>

<script>
let equityChart=null,ddChart=null,pollInterval=null,currentPage=0,currentMode='backtest';
const PS=50;

function $(s){return document.getElementById(s)}
function money(v){return '$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:0})}
function pct(v){return(v*100).toFixed(2)+'%'}
function cls(v){return v>=0?'pos':'neg'}
function num(v,d=2){return Number(v).toFixed(d)}
function switchTab(n,el){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  $('tab-'+n).classList.add('active');el.classList.add('active');
  if(n==='trades')loadTrades(0);
}

function setMode(m){
  currentMode=m;
  $('mode-bt').classList.toggle('active',m==='backtest');
  $('mode-live').classList.toggle('active',m==='live');
  $('interval-sel').style.display=m==='live'?'inline-block':'none';
  $('live-bar').style.display=m==='live'?'grid':'none';
  $('prog-fill').className='prog-fill '+(m==='live'?'prog-fill-live':'prog-fill-bt');
  $('chart-title').textContent=m==='live'?'Live Equity':'Equity Curve';
}

function initCharts(){
  equityChart=new Chart($('equityChart').getContext('2d'),{type:'line',
    data:{labels:[],datasets:[{label:'Equity',data:[],borderColor:'#10b981',backgroundColor:'rgba(16,185,129,.08)',borderWidth:2,pointRadius:0,fill:true,tension:.1}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},
      scales:{x:{ticks:{maxTicksLimit:10,color:'#5a6a84',font:{size:9}},grid:{color:'rgba(30,45,69,.3)'}},
              y:{ticks:{color:'#94a3b8',font:{size:10},callback:v=>money(v)},grid:{color:'rgba(30,45,69,.3)'}}},
      plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false}}}});
  ddChart=new Chart($('ddChart').getContext('2d'),{type:'line',
    data:{labels:[],datasets:[{label:'Drawdown',data:[],borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,.15)',borderWidth:1.5,pointRadius:0,fill:true,tension:.1}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},
      scales:{x:{ticks:{maxTicksLimit:8,color:'#5a6a84',font:{size:9}},grid:{color:'rgba(30,45,69,.3)'}},
              y:{ticks:{color:'#94a3b8',font:{size:10},callback:v=>pct(v)},grid:{color:'rgba(30,45,69,.3)'}}},
      plugins:{legend:{display:false}}}});
}

async function doStart(){
  $('btn-start').disabled=true;
  const body={mode:currentMode};
  if(currentMode==='live') body.interval=parseInt($('interval-sel').value);
  try{
    await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    startPolling();
  }catch(e){alert('Failed: '+e);$('btn-start').disabled=false}
}
async function doStop(){
  await fetch('/api/stop',{method:'POST'});
  $('btn-stop').style.display='none';
  $('btn-rebal').style.display='none';
}
async function doRebalance(){
  $('btn-rebal').disabled=true;
  await fetch('/api/force-rebalance',{method:'POST'});
  setTimeout(()=>{$('btn-rebal').disabled=false},3000);
}
async function doReset(){
  await fetch('/api/reset',{method:'POST'});
  stopPolling();
  $('btn-start').disabled=false;$('btn-stop').style.display='none';$('btn-rebal').style.display='none';
  $('status-badge').className='badge badge-idle';$('status-badge').textContent='IDLE';
  $('port-val').textContent='$1,000,000';$('port-chg').textContent='+0.00%';$('port-chg').className='port-chg';
  $('prog-fill').style.width='0%';$('prog-label').textContent='Press START to begin';$('elapsed').textContent='';
  ['s-ann-ret','s-sharpe','s-sortino','s-max-dd','s-win-rate','s-pf','s-calmar'].forEach(i=>$(i).textContent='—');
  $('s-trades').textContent='0';
  $('metrics-body').innerHTML='<div class="empty">Run a backtest or start live trading</div>';
  $('model-weights-body').innerHTML='<div class="empty">Waiting for data...</div>';
  $('trades-body').innerHTML='<tr><td colspan="6" class="empty">No trades yet</td></tr>';
  $('trade-count').textContent='0 trades';
  $('live-bar').style.display='none';
  if(equityChart){equityChart.data.labels=[];equityChart.data.datasets[0].data=[];equityChart.update()}
  if(ddChart){ddChart.data.labels=[];ddChart.data.datasets[0].data=[];ddChart.update()}
}

function startPolling(){if(!pollInterval)pollInterval=setInterval(pollStatus,currentMode==='live'?3000:800)}
function stopPolling(){if(pollInterval){clearInterval(pollInterval);pollInterval=null}}

async function pollStatus(){
  try{
    const d=await(await fetch('/api/status')).json();
    updateUI(d);
    if(d.status==='complete'||d.status==='error'){
      stopPolling();$('btn-start').disabled=false;
      if(d.status==='complete')loadFinalResults();
    }
    if(d.status==='live_stopped'){stopPolling();$('btn-start').disabled=false;$('btn-stop').style.display='none';$('btn-rebal').style.display='none'}
  }catch(e){console.error(e)}
}

function updateUI(d){
  // Badge
  const bm={idle:'badge-idle',loading:'badge-running',running:'badge-running',complete:'badge-complete',
            error:'badge-error',live_active:'badge-live',live_stopped:'badge-idle'};
  $('status-badge').className='badge '+(bm[d.status]||'badge-idle');
  const labels={idle:'IDLE',loading:'LOADING',running:'RUNNING',complete:'COMPLETE',error:'ERROR',
                live_active:'LIVE',live_stopped:'STOPPED'};
  $('status-badge').textContent=labels[d.status]||d.status;

  // Live controls
  if(d.status==='live_active'){
    $('btn-stop').style.display='inline-block';$('btn-rebal').style.display='inline-block';
    $('btn-start').disabled=true;
    setMode('live');
  }

  if(d.elapsed>0)$('elapsed').textContent=num(d.elapsed,0)+'s';

  // Progress
  $('prog-fill').style.width=(d.mode==='live'?100:d.progress)+'%';
  $('prog-label').textContent=d.progress_msg||(d.status==='running'?`Day ${d.current_day}/${d.total_days}`:'');

  // Equity
  const eq=d.equity_history;
  if(eq&&eq.length>0){
    const val=eq[eq.length-1].equity;
    $('port-val').textContent=money(val);
    const tr=d.metrics?.total_return||0;
    $('port-chg').textContent=(tr>=0?'+':'')+pct(tr);
    $('port-chg').className='port-chg '+cls(tr);
    equityChart.data.labels=eq.map(e=>e.date);
    equityChart.data.datasets[0].data=eq.map(e=>e.equity);
    equityChart.data.datasets[0].borderColor=d.mode==='live'?'#fb923c':'#10b981';
    equityChart.data.datasets[0].backgroundColor=d.mode==='live'?'rgba(251,146,60,.08)':'rgba(16,185,129,.08)';
    equityChart.update();
    let peak=eq[0].equity;
    ddChart.data.labels=eq.map(e=>e.date);
    ddChart.data.datasets[0].data=eq.map(e=>{peak=Math.max(peak,e.equity);return(e.equity-peak)/peak});
    ddChart.update();
    $('chart-info').textContent=eq.length+(d.mode==='live'?' snapshots':' days');
  }

  // Metrics
  const m=d.metrics;
  if(m){
    $('s-ann-ret').textContent=pct(m.ann_return);$('s-ann-ret').className='stat-v '+cls(m.ann_return);
    $('s-sharpe').textContent=num(m.sharpe);$('s-sharpe').className='stat-v '+cls(m.sharpe);
    $('s-sortino').textContent=num(m.sortino);
    $('s-max-dd').textContent=pct(m.max_dd);$('s-max-dd').className='stat-v neg';
    $('s-win-rate').textContent=pct(m.win_rate);
    $('s-pf').textContent=num(m.profit_factor);
    $('s-calmar').textContent=num(m.calmar);
    $('s-trades').textContent=d.trade_count||0;
    $('metrics-body').innerHTML=[
      ['Total Return',pct(m.total_return),cls(m.total_return)],
      ['Ann. Return',pct(m.ann_return),cls(m.ann_return)],
      ['Ann. Volatility',pct(m.ann_vol),''],
      ['Sharpe',num(m.sharpe),cls(m.sharpe)],
      ['Sortino',num(m.sortino),cls(m.sortino)],
      ['Max Drawdown',pct(m.max_dd),'neg'],
      ['Win Rate',pct(m.win_rate),m.win_rate>0.5?'pos':''],
      ['Profit Factor',num(m.profit_factor),m.profit_factor>1?'pos':'neg'],
      ['Calmar',num(m.calmar),cls(m.calmar)],
    ].map(([l,v,c])=>`<div class="m-row"><span class="m-label">${l}</span><span class="m-val ${c}">${v}</span></div>`).join('');
  }

  // Model weights
  const mw=d.model_weights;
  if(mw&&Object.keys(mw).length){
    const sorted=Object.entries(mw).sort((a,b)=>b[1]-a[1]);
    const mx=Math.max(...sorted.map(([,v])=>v),.01);
    const colors=['#10b981','#3b82f6','#a78bfa','#22d3ee','#fb923c','#ec4899'];
    $('model-weights-body').innerHTML=sorted.map(([n,w],i)=>{
      return`<div class="w-bar-wrap"><div class="w-bar-label"><span>${n}</span><span>${(w*100).toFixed(1)}%</span></div>
        <div class="w-bar-bg"><div class="w-bar-fill" style="width:${(w/mx*100).toFixed(0)}%;background:${colors[i%6]}"></div></div></div>`;
    }).join('');
  }

  // Live account info
  if(d.mode==='live'&&d.live_account&&d.live_account.equity){
    $('live-bar').style.display='grid';
    $('la-equity').textContent=money(d.live_account.equity);
    $('la-cash').textContent=money(d.live_account.cash);
    $('la-bp').textContent=money(d.live_account.buying_power);
    $('la-market').innerHTML=d.market_open?'<span class="pos">OPEN</span>':'<span class="neg">CLOSED</span>';
    $('la-last-rb').textContent=d.last_rebalance||'—';
    $('la-next-rb').textContent=d.next_rebalance||'—';
    $('la-rb-count').textContent=d.rebalance_count;
    $('la-pos-count').textContent=(d.live_positions||[]).length;
  }

  // Positions tab
  const lp=d.live_positions;
  if(lp&&lp.length>0){
    $('pos-count').textContent=lp.length+' positions';
    $('positions-body').innerHTML='<table class="pos-table"><thead><tr><th>Symbol</th><th>Qty</th><th>Price</th><th>Entry</th><th>Value</th><th>P&L</th><th>P&L%</th></tr></thead><tbody>'+
      lp.map(p=>{const c=p.unrealized_pl>=0?'pos':'neg';
        return`<tr><td style="font-weight:800">${p.symbol}</td><td>${p.qty}</td><td>${money(p.current_price)}</td><td>${money(p.avg_entry_price)}</td><td>${money(p.market_value)}</td><td class="${c}">${money(p.unrealized_pl)}</td><td class="${c}">${num(p.unrealized_plpc)}%</td></tr>`}).join('')+
      '</tbody></table>';
  }

  // Weights/risk tabs
  const cw=d.current_weights;
  if(cw&&Object.keys(cw).length){
    $('weights-body').innerHTML='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:6px">'+
      Object.entries(cw).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(([t,w])=>
        `<div style="background:var(--bg1);padding:8px;border-radius:4px;border:1px solid var(--border)">
          <div style="font-weight:800;font-size:12px">${t}</div>
          <div class="${cls(w)}" style="font-size:14px;font-weight:700">${pct(w)}</div></div>`).join('')+'</div>';
  }
  const rm=d.risk_metrics;
  if(rm&&Object.keys(rm).length){
    $('risk-body').innerHTML=Object.entries(rm).map(([k,v])=>{
      const val=typeof v==='number'?(Math.abs(v)<1?pct(v):num(v,4)):String(v);
      return`<div class="m-row"><span class="m-label">${k}</span><span class="m-val">${val}</span></div>`;}).join('');
  }
}

async function loadFinalResults(){
  try{const d=await(await fetch('/api/result')).json();
    if(d.equity_curve){equityChart.data.labels=d.equity_curve.dates;equityChart.data.datasets[0].data=d.equity_curve.values;equityChart.update()}
    if(d.drawdown){ddChart.data.labels=d.equity_curve?.dates||[];ddChart.data.datasets[0].data=d.drawdown;ddChart.update()}
    const p=d.performance;
    if(p){
      $('port-val').textContent=money(p.final_equity);
      $('port-chg').textContent=(p.total_return>=0?'+':'')+pct(p.total_return);
      $('port-chg').className='port-chg '+cls(p.total_return);
      $('s-ann-ret').textContent=pct(p.annualized_return);$('s-sharpe').textContent=num(p.sharpe_ratio);
      $('s-sortino').textContent=num(p.sortino_ratio);$('s-max-dd').textContent=pct(p.max_drawdown);
      $('s-win-rate').textContent=pct(p.win_rate);$('s-pf').textContent=num(p.profit_factor);
      $('s-calmar').textContent=num(p.calmar_ratio);$('s-trades').textContent=d.detailed_trades?.length||0;
      $('metrics-body').innerHTML=[
        ['Total Return',pct(p.total_return),cls(p.total_return)],['Ann. Return',pct(p.annualized_return),cls(p.annualized_return)],
        ['Ann. Vol',pct(p.annualized_volatility),''],['Sharpe',num(p.sharpe_ratio),cls(p.sharpe_ratio)],
        ['Sortino',num(p.sortino_ratio),cls(p.sortino_ratio)],['Max DD',pct(p.max_drawdown),'neg'],
        ['Calmar',num(p.calmar_ratio),cls(p.calmar_ratio)],['Win Rate',pct(p.win_rate),p.win_rate>0.5?'pos':''],
        ['Profit Factor',num(p.profit_factor),p.profit_factor>1?'pos':'neg'],
        ['Skewness',num(p.skewness),''],['Kurtosis',num(p.kurtosis),''],['Trading Days',p.n_trading_days,''],
      ].map(([l,v,c])=>`<div class="m-row"><span class="m-label">${l}</span><span class="m-val ${c}">${v}</span></div>`).join('');
      if(d.execution){
        $('metrics-body').innerHTML+='<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">'+
          [['Total Costs',money(d.execution.total_costs),'neg'],['Turnover',num(d.execution.total_turnover)+'x',''],
           ['Rebalances',d.execution.n_rebalances,''],['Avg Cost/Rebal',money(d.execution.avg_cost_per_rebalance),'']
          ].map(([l,v,c])=>`<div class="m-row"><span class="m-label">${l}</span><span class="m-val ${c||''}">${v}</span></div>`).join('')+'</div>';
      }
    }
    loadTrades(0);
  }catch(e){console.error(e)}
}

async function loadTrades(page){
  currentPage=page;
  try{const d=await(await fetch(`/api/trades?page=${page}&size=${PS}`)).json();
    $('trade-count').textContent=d.total+' trades';
    if(!d.trades||!d.trades.length){
      $('trades-thead').innerHTML='';$('trades-body').innerHTML='<tr><td colspan="6" class="empty">No trades yet</td></tr>';
      $('page-controls').innerHTML='';return;
    }
    const isLive=d.mode==='live';
    if(isLive){
      $('trades-thead').innerHTML='<tr><th>#</th><th>Time</th><th>Symbol</th><th>Side</th><th>Amount</th><th>Status</th></tr>';
      $('trades-body').innerHTML=d.trades.map((t,i)=>{
        const idx=d.total-page*PS-i;
        const sideC=t.side==='buy'||t.side==='BUY'?'pos':t.side==='sell'||t.side==='SELL'?'neg':'';
        return`<tr><td style="color:var(--text3)">${idx}</td><td>${t.time||''}</td><td style="font-weight:800">${t.symbol||''}</td>
          <td class="${sideC}" style="font-weight:700">${(t.side||'').toUpperCase()}</td>
          <td>${t.notional?money(t.notional):''}</td><td>${t.status||''}</td></tr>`;}).join('');
    }else{
      $('trades-thead').innerHTML='<tr><th>#</th><th>Date</th><th>Trades</th><th>Turnover</th><th>Cost</th><th>Portfolio</th></tr>';
      $('trades-body').innerHTML=d.trades.map((t,i)=>{
        const idx=d.total-page*PS-i;
        return`<tr><td style="color:var(--text3)">${idx}</td><td>${t.date}</td><td>${t.n_trades}</td>
          <td>${num((t.turnover||0)*100,1)}%</td><td class="${t.cost>500?'neg':''}">${money(t.cost||0)}</td>
          <td>${money(t.port_value||0)}</td></tr>`;}).join('');
    }
    if(d.pages>1){
      let b='';for(let p=0;p<Math.min(d.pages,10);p++)b+=`<button class="page-btn ${p===page?'active':''}" onclick="loadTrades(${p})">${p+1}</button>`;
      if(d.pages>10)b+=`<span style="color:var(--text3);padding:4px">...${d.pages}</span>`;
      $('page-controls').innerHTML=b;
    }else $('page-controls').innerHTML='';
  }catch(e){console.error(e)}
}

initCharts();
(async()=>{try{const d=await(await fetch('/api/status')).json();
  if(d.status==='running'||d.status==='loading'){$('btn-start').disabled=true;startPolling()}
  else if(d.status==='complete'){updateUI(d);loadFinalResults()}
  else if(d.status==='live_active'){setMode('live');$('btn-start').disabled=true;startPolling()}
}catch(e){}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Trading Dashboard")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Quant Trading Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Modes: Backtest | Live Paper Trading\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
