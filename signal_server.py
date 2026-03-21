"""
Signal Server — Generates quant engine signals and serves them via REST API.

This server runs the quant engine models on historical data and provides
real-time signals that the paper-trader-v4.html can consume.

Usage:
    python signal_server.py                     # Start on port 5555
    python signal_server.py --port 5555         # Custom port
    python signal_server.py --refresh 300       # Refresh signals every 5 min

Endpoints:
    GET /api/signals         — Current alpha signals for all tickers
    GET /api/weights          — Target portfolio weights
    GET /api/brain            — Brain export for paper-trader-v4
    GET /api/performance      — Latest backtest performance metrics
    GET /api/prices           — Latest prices from Alpaca
    GET /api/historical       — Historical price data (for charting)
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
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
logger = logging.getLogger("SignalServer")

app = Flask(__name__)
CORS(app)

# ── Global state ──────────────────────────────────────────────
signal_state = {
    "status": "idle",
    "last_update": None,
    "signals": {},           # ticker -> signal strength
    "weights": {},           # ticker -> target weight
    "model_weights": {},     # model -> weight
    "performance": {},       # backtest metrics
    "brain_export": None,    # for paper-trader-v4
    "prices": {},            # latest prices
    "historical": {},        # historical price series
    "risk_metrics": {},
    "refresh_interval": 300,
}
state_lock = threading.Lock()


def _build_models(config):
    """Build alpha models."""
    alpha_cfg = config.alpha
    models = []
    for name in alpha_cfg.enabled_models:
        if name == "ts_momentum": models.append(TimeSeriesMomentum(alpha_cfg))
        elif name == "xs_momentum": models.append(CrossSectionalMomentum(alpha_cfg))
        elif name == "momentum_vol_break": models.append(MomentumWithVolBreak(alpha_cfg))
        elif name == "ou_mean_reversion": models.append(OUMeanReversion(alpha_cfg))
        elif name == "pairs_trading": models.append(PairsTrading(alpha_cfg))
        elif name == "ml_alpha": models.append(MLAlpha(alpha_cfg))
    return models


def _refresh_signals():
    """Run quant engine pipeline and update signals."""
    try:
        with state_lock:
            signal_state["status"] = "refreshing"

        config = SystemConfig()
        logger.info("Refreshing signals — fetching data...")

        # Fetch historical data
        feed = DataFeed(config.data)
        feed.load()
        logger.info(f"Data loaded: {feed.prices.shape}")

        # Build models
        models = _build_models(config)

        # Compute features
        feature_engine = FeatureEngine(config.features)
        features = feature_engine.generate(feed)

        # Generate signals
        all_signals = {}
        for model in models:
            try:
                sig = model.generate_signals(feed, features)
                all_signals[model.name] = sig
            except Exception as e:
                logger.warning(f"Model {model.name} failed: {e}")

        # Aggregate
        aggregator = SignalAggregator(
            method=config.alpha.ensemble_method,
            ic_lookback=63,
        )
        AGG_LOOKBACK = 300
        returns = feed.returns
        ret_end = len(returns)
        ret_start = max(0, ret_end - AGG_LOOKBACK)
        recent_returns = returns.iloc[ret_start:ret_end]

        date_signals = {}
        for name, sig_df in all_signals.items():
            date_idx = len(sig_df) - 1
            start_idx = max(0, date_idx - AGG_LOOKBACK + 1)
            date_signals[name] = sig_df.iloc[start_idx:date_idx + 1]

        combined = aggregator.aggregate(date_signals, recent_returns)
        signal_row = combined.iloc[-1]

        # Portfolio optimization
        optimizer = PortfolioOptimizer(config.portfolio)
        lookback_returns = returns.iloc[-252:]
        cov_matrix = PortfolioOptimizer.estimate_covariance(
            lookback_returns, method="exponential", halflife=63
        )
        current_weights = pd.Series(0.0, index=signal_row.index)
        target_weights = optimizer.optimize(signal_row, cov_matrix, current_weights)

        # Risk management
        risk_mgr = RiskManager(config.risk)
        equity_curve = pd.Series([1_000_000], index=[pd.Timestamp.now()])
        target_weights = risk_mgr.check_and_adjust(target_weights, lookback_returns, equity_curve)

        # Run full backtest for performance metrics
        engine = BacktestEngine(config)
        result = engine.run(feed, models, initial_capital=1_000_000)

        # Export brain
        try:
            brain_data = engine.export_brain(result)
        except Exception:
            brain_data = None

        # Extract latest prices
        latest_prices = feed.prices.iloc[-1].to_dict()

        # Historical prices for charting (last 252 days, downsampled)
        historical = {}
        for col in feed.prices.columns:
            historical[col] = {
                "dates": [str(d)[:10] for d in feed.prices.index[-252:]],
                "prices": feed.prices[col].iloc[-252:].round(2).tolist(),
            }

        with state_lock:
            signal_state["status"] = "ready"
            signal_state["last_update"] = datetime.now().isoformat()
            signal_state["signals"] = {
                k: round(float(v), 6)
                for k, v in signal_row.items()
                if abs(v) > 0.001
            }
            signal_state["weights"] = {
                k: round(float(v), 6)
                for k, v in target_weights.items()
                if abs(v) > 0.001
            }
            signal_state["model_weights"] = {
                k: round(float(v), 6)
                for k, v in aggregator.model_weights.items()
            }
            signal_state["performance"] = {
                k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                for k, v in result.performance_metrics.items()
            }
            signal_state["brain_export"] = brain_data
            signal_state["prices"] = {k: round(v, 2) for k, v in latest_prices.items()}
            signal_state["historical"] = historical
            signal_state["risk_metrics"] = risk_mgr.get_risk_report()

        logger.info(f"Signals refreshed. Top signals: {dict(list(sorted(signal_state['signals'].items(), key=lambda x: -abs(x[1]))[:5]))}")

    except Exception as e:
        logger.exception(f"Signal refresh failed: {e}")
        with state_lock:
            signal_state["status"] = "error"


def _signal_refresh_loop(interval):
    """Background loop to refresh signals periodically."""
    while True:
        _refresh_signals()
        logger.info(f"Next refresh in {interval}s")
        time.sleep(interval)


# ── Routes ────────────────────────────────────────────────────

@app.route("/api/signals")
def api_signals():
    with state_lock:
        return jsonify({
            "status": signal_state["status"],
            "last_update": signal_state["last_update"],
            "signals": signal_state["signals"],
            "model_weights": signal_state["model_weights"],
        })


@app.route("/api/weights")
def api_weights():
    with state_lock:
        return jsonify({
            "status": signal_state["status"],
            "last_update": signal_state["last_update"],
            "weights": signal_state["weights"],
            "risk_metrics": signal_state["risk_metrics"],
        })


@app.route("/api/brain")
def api_brain():
    with state_lock:
        if signal_state["brain_export"]:
            return jsonify(signal_state["brain_export"])
        return jsonify({"error": "No brain data available"}), 404


@app.route("/api/performance")
def api_performance():
    with state_lock:
        return jsonify({
            "status": signal_state["status"],
            "last_update": signal_state["last_update"],
            "performance": signal_state["performance"],
        })


@app.route("/api/prices")
def api_prices():
    with state_lock:
        return jsonify({
            "prices": signal_state["prices"],
            "last_update": signal_state["last_update"],
        })


@app.route("/api/historical")
def api_historical():
    ticker = request.args.get("ticker", "SPY")
    with state_lock:
        if ticker in signal_state["historical"]:
            return jsonify(signal_state["historical"][ticker])
        return jsonify({"error": f"No data for {ticker}"}), 404


@app.route("/api/historical/all")
def api_historical_all():
    with state_lock:
        return jsonify({
            "tickers": list(signal_state["historical"].keys()),
            "last_update": signal_state["last_update"],
        })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    threading.Thread(target=_refresh_signals, daemon=True).start()
    return jsonify({"status": "refresh_triggered"})


@app.route("/")
def index():
    return jsonify({
        "name": "Quant Engine Signal Server",
        "version": "1.0",
        "endpoints": [
            "/api/signals", "/api/weights", "/api/brain",
            "/api/performance", "/api/prices", "/api/historical?ticker=SPY",
            "/api/historical/all", "/api/refresh (POST)",
        ],
        "status": signal_state["status"],
        "last_update": signal_state["last_update"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Engine Signal Server")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--refresh", type=int, default=300,
                        help="Signal refresh interval in seconds (default: 300)")
    args = parser.parse_args()

    signal_state["refresh_interval"] = args.refresh

    # Start signal refresh loop in background
    refresh_thread = threading.Thread(
        target=_signal_refresh_loop,
        args=(args.refresh,),
        daemon=True,
    )
    refresh_thread.start()

    print(f"\n  Quant Engine Signal Server")
    print(f"  API:       http://localhost:{args.port}")
    print(f"  Signals:   http://localhost:{args.port}/api/signals")
    print(f"  Brain:     http://localhost:{args.port}/api/brain")
    print(f"  Refresh:   every {args.refresh}s\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
