"""
Live Trading Bot — Bridges Quant Engine signals to Alpaca API.

Uses the quant-engine-local models for signal generation,
then executes trades via Alpaca paper trading API.

Supports:
    - Real-time trading with live market data
    - Historical backtesting via yfinance
    - WebSocket streaming for live price updates
    - Automatic rebalancing based on quant engine signals

Usage:
    python live_trader.py                        # Start live trading (paper)
    python live_trader.py --backtest             # Run backtest only
    python live_trader.py --interval 300         # Rebalance every 5 min
    python live_trader.py --port 8080            # Web dashboard on port 8080
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

# Add quant engine to path
QUANT_ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "quant-engine-local")
sys.path.insert(0, os.path.abspath(QUANT_ENGINE_PATH))

from alpha.mean_reversion import OUMeanReversion, PairsTrading
from alpha.ml_alpha import MLAlpha
from alpha.momentum import (
    CrossSectionalMomentum,
    MomentumWithVolBreak,
    TimeSeriesMomentum,
)
from config import SystemConfig
from data.feed import DataFeed
from ensemble.aggregator import SignalAggregator
from features.engine import FeatureEngine
from portfolio.optimizer import PortfolioOptimizer
from risk.manager import RiskManager

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveTrader")

# ── Configuration ─────────────────────────────────────────────
ALPACA_KEY = os.environ.get("ALPACA_KEY", "PKA4ZJAUHC2XSXBMVFISRR7H2Q")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "FnbWgaFKFzqEkEoroSA9ioXPJUTZm1SGyM894pUfH1dE")
PAPER = True  # Always paper trading


class LiveTrader:
    """
    Connects quant engine alpha models to Alpaca API for live paper trading.

    Architecture:
        1. Fetch historical data (yfinance) for model training/warmup
        2. Generate signals using all 6 quant engine models
        3. Aggregate signals via performance-weighted ensemble
        4. Optimize portfolio via mean-variance
        5. Apply risk constraints
        6. Execute trades via Alpaca API
        7. Repeat at configured interval
    """

    def __init__(self, config: SystemConfig = None, rebalance_interval: int = 600):
        self.config = config or SystemConfig()
        self.rebalance_interval = rebalance_interval  # seconds

        # Alpaca clients
        self.trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
        self.data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

        # Quant engine components
        self.feature_engine = FeatureEngine(self.config.features)
        self.signal_aggregator = SignalAggregator(
            method=self.config.alpha.ensemble_method,
            ic_lookback=63,
        )
        self.portfolio_optimizer = PortfolioOptimizer(self.config.portfolio)
        self.risk_manager = RiskManager(self.config.risk)

        # Alpha models
        self.models = self._build_models()

        # State
        self.running = False
        self.last_rebalance = None
        self.trade_history = []
        self.equity_history = []
        self.current_signals = {}
        self.current_weights = {}
        self.live_prices = {}
        self.status = "idle"

        # Filter to stock tickers only (no ETF issues with Alpaca)
        self.tradeable_tickers = self._get_tradeable_tickers()

    def _build_models(self):
        """Build all alpha models from config."""
        alpha_cfg = self.config.alpha
        models = []
        for name in alpha_cfg.enabled_models:
            if name == "ts_momentum":
                models.append(TimeSeriesMomentum(alpha_cfg))
            elif name == "xs_momentum":
                models.append(CrossSectionalMomentum(alpha_cfg))
            elif name == "momentum_vol_break":
                models.append(MomentumWithVolBreak(alpha_cfg))
            elif name == "ou_mean_reversion":
                models.append(OUMeanReversion(alpha_cfg))
            elif name == "pairs_trading":
                models.append(PairsTrading(alpha_cfg))
            elif name == "ml_alpha":
                models.append(MLAlpha(alpha_cfg))
        return models

    def _get_tradeable_tickers(self):
        """Filter tickers to ones tradeable on Alpaca."""
        try:
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca. Account equity: ${float(account.equity):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return self.config.data.tickers

        tradeable = []
        for ticker in self.config.data.tickers:
            try:
                asset = self.trading_client.get_asset(ticker)
                if asset.tradable:
                    tradeable.append(ticker)
            except Exception:
                logger.warning(f"Ticker {ticker} not found on Alpaca, skipping")
        logger.info(f"Tradeable tickers: {len(tradeable)}/{len(self.config.data.tickers)}")
        return tradeable

    def fetch_historical_data(self):
        """Fetch historical data using yfinance (for model training)."""
        logger.info("Fetching historical data for model training...")
        self.status = "loading_data"
        data_config = self.config.data
        data_config.tickers = self.tradeable_tickers
        feed = DataFeed(data_config)
        feed.load()
        logger.info(f"Historical data: {feed.prices.shape[0]} days x {feed.prices.shape[1]} assets")
        return feed

    def generate_signals(self, data: DataFeed):
        """Run all alpha models and aggregate signals."""
        logger.info("Generating alpha signals...")
        self.status = "generating_signals"

        # Compute features
        features = self.feature_engine.generate(data)

        # Generate signals from each model
        all_signals = {}
        for model in self.models:
            try:
                sig = model.generate_signals(data, features)
                all_signals[model.name] = sig
                logger.info(f"  {model.name}: signal range [{sig.min().min():.3f}, {sig.max().max():.3f}]")
            except Exception as e:
                logger.warning(f"  {model.name} failed: {e}")

        if not all_signals:
            logger.error("No signals generated!")
            return None

        # Aggregate
        returns = data.returns
        AGG_LOOKBACK = 300
        ret_end = len(returns)
        ret_start = max(0, ret_end - AGG_LOOKBACK)
        recent_returns = returns.iloc[ret_start:ret_end]

        date_signals = {}
        for name, sig_df in all_signals.items():
            date_idx = len(sig_df) - 1
            start_idx = max(0, date_idx - AGG_LOOKBACK + 1)
            date_signals[name] = sig_df.iloc[start_idx:date_idx + 1]

        try:
            combined = self.signal_aggregator.aggregate(date_signals, recent_returns)
            signal_row = combined.iloc[-1]
            self.current_signals = signal_row.to_dict()
            logger.info(f"Combined signal: top={signal_row.nlargest(3).to_dict()}")
            return signal_row, returns
        except Exception as e:
            logger.error(f"Signal aggregation failed: {e}")
            return None

    def compute_target_weights(self, signal_row, returns):
        """Compute optimal portfolio weights from signals."""
        logger.info("Computing target portfolio weights...")
        self.status = "optimizing"

        lookback_returns = returns.iloc[-252:]
        if len(lookback_returns) < 60:
            logger.error("Insufficient data for optimization")
            return None

        # Covariance estimation
        cov_matrix = PortfolioOptimizer.estimate_covariance(
            lookback_returns, method="exponential", halflife=63
        )

        # Current weights (from Alpaca positions)
        current_weights = self._get_current_weights()

        # Optimize
        try:
            target_weights = self.portfolio_optimizer.optimize(
                signal_row, cov_matrix, current_weights
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

        # Risk adjustments
        equity_series = pd.Series(
            [h["equity"] for h in self.equity_history] or [self._get_equity()],
            index=pd.date_range(end=pd.Timestamp.now(), periods=max(len(self.equity_history), 1))
        )
        target_weights = self.risk_manager.check_and_adjust(
            target_weights, lookback_returns, equity_series
        )

        self.current_weights = {k: round(v, 4) for k, v in target_weights.items() if abs(v) > 0.001}
        logger.info(f"Target weights: {self.current_weights}")
        return target_weights

    def execute_trades(self, target_weights):
        """Execute trades via Alpaca API to reach target weights."""
        logger.info("Executing trades...")
        self.status = "trading"

        account = self.trading_client.get_account()
        portfolio_value = float(account.equity)

        # Get current positions
        positions = self.trading_client.get_all_positions()
        current_positions = {}
        for pos in positions:
            current_positions[pos.symbol] = {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "current_price": float(pos.current_price),
            }

        trades_executed = []

        for ticker, target_weight in target_weights.items():
            if abs(target_weight) < 0.001:
                # Close position if exists
                if ticker in current_positions:
                    try:
                        self.trading_client.close_position(ticker)
                        trades_executed.append({
                            "ticker": ticker, "action": "CLOSE",
                            "time": datetime.now().isoformat(),
                        })
                        logger.info(f"  CLOSE {ticker}")
                    except Exception as e:
                        logger.warning(f"  Failed to close {ticker}: {e}")
                continue

            # Target dollar value
            target_value = portfolio_value * target_weight
            current_value = current_positions.get(ticker, {}).get("market_value", 0)
            trade_value = target_value - current_value

            # Skip small trades (< 0.5% of portfolio)
            if abs(trade_value) < portfolio_value * 0.005:
                continue

            # Determine order
            side = OrderSide.BUY if trade_value > 0 else OrderSide.SELL
            notional = abs(trade_value)

            try:
                order_request = MarketOrderRequest(
                    symbol=ticker,
                    notional=round(notional, 2),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_request)
                trades_executed.append({
                    "ticker": ticker,
                    "action": side.value,
                    "notional": round(notional, 2),
                    "weight": round(target_weight, 4),
                    "order_id": str(order.id),
                    "time": datetime.now().isoformat(),
                })
                logger.info(f"  {side.value} ${notional:,.0f} of {ticker} (target weight: {target_weight:.2%})")
            except Exception as e:
                logger.warning(f"  Failed to trade {ticker}: {e}")

        self.trade_history.extend(trades_executed)
        logger.info(f"Executed {len(trades_executed)} trades")
        return trades_executed

    def _get_equity(self):
        """Get current portfolio equity from Alpaca."""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception:
            return 0

    def _get_current_weights(self):
        """Get current portfolio weights from Alpaca positions."""
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            positions = self.trading_client.get_all_positions()
            weights = pd.Series(0.0, index=self.tradeable_tickers)
            for pos in positions:
                if pos.symbol in weights.index:
                    weights[pos.symbol] = float(pos.market_value) / equity
            return weights
        except Exception:
            return pd.Series(0.0, index=self.tradeable_tickers)

    def rebalance(self):
        """Full rebalance cycle: data -> signals -> optimize -> trade."""
        logger.info("=" * 60)
        logger.info("REBALANCE CYCLE")
        logger.info("=" * 60)

        try:
            # 1. Fetch historical data
            data = self.fetch_historical_data()

            # 2. Generate signals
            result = self.generate_signals(data)
            if result is None:
                return

            signal_row, returns = result

            # 3. Compute target weights
            target_weights = self.compute_target_weights(signal_row, returns)
            if target_weights is None:
                return

            # 4. Execute trades
            trades = self.execute_trades(target_weights)

            # 5. Record state
            equity = self._get_equity()
            self.equity_history.append({
                "time": datetime.now().isoformat(),
                "equity": equity,
            })

            self.last_rebalance = datetime.now()
            self.status = "active"

            logger.info(f"Rebalance complete. Equity: ${equity:,.2f}")
            logger.info(f"Model weights: {self.signal_aggregator.model_weights}")
            logger.info("=" * 60)

        except Exception as e:
            logger.exception(f"Rebalance failed: {e}")
            self.status = "error"

    def run(self):
        """Main trading loop — rebalance at configured interval."""
        self.running = True
        self.status = "starting"
        logger.info(f"Live Trader starting (interval: {self.rebalance_interval}s)")

        # Initial rebalance
        self.rebalance()

        while self.running:
            try:
                time.sleep(self.rebalance_interval)
                if not self.running:
                    break

                # Check if market is open
                clock = self.trading_client.get_clock()
                if clock.is_open:
                    self.rebalance()
                else:
                    next_open = clock.next_open
                    logger.info(f"Market closed. Next open: {next_open}")
                    self.status = "market_closed"
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(60)

        self.status = "stopped"
        logger.info("Live Trader stopped")

    def stop(self):
        """Stop the trading loop."""
        self.running = False

    def get_status(self):
        """Get current trader status for API/dashboard."""
        return {
            "status": self.status,
            "running": self.running,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "rebalance_interval": self.rebalance_interval,
            "equity": self._get_equity(),
            "equity_history": self.equity_history[-100:],
            "current_weights": self.current_weights,
            "current_signals": {k: round(v, 4) for k, v in self.current_signals.items() if abs(v) > 0.01},
            "model_weights": self.signal_aggregator.model_weights,
            "risk_metrics": self.risk_manager.get_risk_report(),
            "recent_trades": self.trade_history[-20:],
            "tradeable_tickers": self.tradeable_tickers,
            "n_models": len(self.models),
        }


# ── Flask API for Web Dashboard ──────────────────────────────
from flask import Flask, jsonify
from flask_cors import CORS

web_app = Flask(__name__)
CORS(web_app)

trader_instance = None


@web_app.route("/api/status")
def api_status():
    if trader_instance:
        return jsonify(trader_instance.get_status())
    return jsonify({"status": "not_initialized"})


@web_app.route("/api/rebalance", methods=["POST"])
def api_rebalance():
    if trader_instance and trader_instance.running:
        threading.Thread(target=trader_instance.rebalance, daemon=True).start()
        return jsonify({"status": "rebalance_triggered"})
    return jsonify({"error": "Trader not running"}), 400


@web_app.route("/api/stop", methods=["POST"])
def api_stop():
    if trader_instance:
        trader_instance.stop()
        return jsonify({"status": "stopping"})
    return jsonify({"error": "Trader not running"}), 400


@web_app.route("/api/positions")
def api_positions():
    if trader_instance:
        try:
            positions = trader_instance.trading_client.get_all_positions()
            return jsonify([{
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            } for p in positions])
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])


@web_app.route("/api/account")
def api_account():
    if trader_instance:
        try:
            account = trader_instance.trading_client.get_account()
            return jsonify({
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Not connected"})


@web_app.route("/api/brain")
def api_brain():
    """Export brain data for paper-trader-v4 integration."""
    if trader_instance:
        brain = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "source": "live-trader",
            "model_weights": trader_instance.signal_aggregator.model_weights,
            "current_signals": trader_instance.current_signals,
            "current_weights": trader_instance.current_weights,
            "risk_metrics": trader_instance.risk_manager.get_risk_report(),
            "equity_history": trader_instance.equity_history[-50:],
        }
        return jsonify(brain)
    return jsonify({"error": "Not initialized"}), 404


@web_app.route("/")
def dashboard():
    return """<!DOCTYPE html>
<html><head><title>Quant Trading Bot - Live Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'SF Mono',monospace;background:#0a0f1a;color:#e8edf5;padding:20px}
h1{color:#d4a017;font-size:20px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px;max-width:1200px}
.card{background:#131c2e;border:1px solid #1e2d45;border-radius:8px;padding:15px}
.card h3{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.metric{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;border-bottom:1px solid #1e2d45}
.metric:last-child{border-bottom:none}
.val{font-weight:700}
.pos{color:#10b981}.neg{color:#f43f5e}
.btn{padding:8px 16px;border:1px solid #1e2d45;background:#1a2540;color:#e8edf5;border-radius:5px;cursor:pointer;font-family:inherit;margin:5px}
.btn:hover{background:#213050}
.btn-green{border-color:#065f46;color:#10b981}
.btn-red{border-color:#881337;color:#f43f5e}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:8px}
.status-active{background:#10b981}
.status-idle{background:#94a3b8}
.status-error{background:#f43f5e}
#positions-table{width:100%;font-size:12px;border-collapse:collapse}
#positions-table th{text-align:left;color:#94a3b8;padding:5px;border-bottom:1px solid #1e2d45}
#positions-table td{padding:5px;border-bottom:1px solid #0c1220}
.full-width{grid-column:1/-1}
#trades-log{max-height:200px;overflow-y:auto;font-size:11px}
.trade-entry{padding:4px 0;border-bottom:1px solid #0c1220}
</style></head><body>
<h1>Quant Trading Bot — Live Dashboard</h1>
<div style="margin-bottom:15px">
<span id="status-indicator"><span class="status-dot status-idle"></span>Initializing...</span>
<button class="btn btn-green" onclick="triggerRebalance()">Force Rebalance</button>
<button class="btn btn-red" onclick="stopTrader()">Stop</button>
</div>
<div class="grid">
<div class="card">
<h3>Account</h3>
<div id="account-info">Loading...</div>
</div>
<div class="card">
<h3>Performance</h3>
<div id="performance-info">Loading...</div>
</div>
<div class="card">
<h3>Model Weights (Ensemble)</h3>
<div id="model-weights">Loading...</div>
</div>
<div class="card">
<h3>Risk Metrics</h3>
<div id="risk-info">Loading...</div>
</div>
<div class="card full-width">
<h3>Current Positions</h3>
<table id="positions-table"><thead><tr><th>Symbol</th><th>Qty</th><th>Value</th><th>P&L</th><th>P&L%</th></tr></thead><tbody id="positions-body"></tbody></table>
</div>
<div class="card">
<h3>Current Signals</h3>
<div id="signals-info">Loading...</div>
</div>
<div class="card">
<h3>Recent Trades</h3>
<div id="trades-log">No trades yet</div>
</div>
</div>
<script>
async function fetchData(){
  try{
    const [status,positions,account]=await Promise.all([
      fetch('/api/status').then(r=>r.json()),
      fetch('/api/positions').then(r=>r.json()),
      fetch('/api/account').then(r=>r.json()),
    ]);
    // Status
    const dot=status.status==='active'?'status-active':status.status==='error'?'status-error':'status-idle';
    document.getElementById('status-indicator').innerHTML=
      `<span class="status-dot ${dot}"></span>${status.status} | Last rebalance: ${status.last_rebalance||'Never'} | Interval: ${status.rebalance_interval}s`;
    // Account
    if(account.equity){
      document.getElementById('account-info').innerHTML=
        `<div class="metric"><span>Equity</span><span class="val">$${Number(account.equity).toLocaleString()}</span></div>`+
        `<div class="metric"><span>Cash</span><span class="val">$${Number(account.cash).toLocaleString()}</span></div>`+
        `<div class="metric"><span>Buying Power</span><span class="val">$${Number(account.buying_power).toLocaleString()}</span></div>`;
    }
    // Model weights
    if(status.model_weights){
      let mw='';
      Object.entries(status.model_weights).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{
        mw+=`<div class="metric"><span>${k}</span><span class="val">${(v*100).toFixed(1)}%</span></div>`;
      });
      document.getElementById('model-weights').innerHTML=mw||'No data';
    }
    // Risk
    if(status.risk_metrics){
      let ri='';
      Object.entries(status.risk_metrics).forEach(([k,v])=>{
        const val=typeof v==='number'?(v*100).toFixed(2)+'%':v;
        ri+=`<div class="metric"><span>${k}</span><span class="val">${val}</span></div>`;
      });
      document.getElementById('risk-info').innerHTML=ri||'No data';
    }
    // Signals
    if(status.current_signals){
      let si='';
      Object.entries(status.current_signals).sort((a,b)=>b[1]-a[1]).slice(0,10).forEach(([k,v])=>{
        const cls=v>0?'pos':'neg';
        si+=`<div class="metric"><span>${k}</span><span class="val ${cls}">${v.toFixed(4)}</span></div>`;
      });
      document.getElementById('signals-info').innerHTML=si||'No signals';
    }
    // Positions
    if(Array.isArray(positions)){
      let pb='';
      positions.forEach(p=>{
        const plCls=p.unrealized_pl>=0?'pos':'neg';
        pb+=`<tr><td>${p.symbol}</td><td>${p.qty}</td><td>$${Number(p.market_value).toLocaleString()}</td>`+
          `<td class="${plCls}">$${Number(p.unrealized_pl).toFixed(2)}</td>`+
          `<td class="${plCls}">${(p.unrealized_plpc*100).toFixed(2)}%</td></tr>`;
      });
      document.getElementById('positions-body').innerHTML=pb||'<tr><td colspan="5">No positions</td></tr>';
    }
    // Trades
    if(status.recent_trades&&status.recent_trades.length){
      let tl='';
      status.recent_trades.slice(-10).reverse().forEach(t=>{
        tl+=`<div class="trade-entry">${t.time?.slice(11,19)||''} ${t.action} ${t.ticker} ${t.notional?'$'+t.notional:''}</div>`;
      });
      document.getElementById('trades-log').innerHTML=tl;
    }
  }catch(e){console.error(e)}
}
async function triggerRebalance(){fetch('/api/rebalance',{method:'POST'})}
async function stopTrader(){fetch('/api/stop',{method:'POST'})}
fetchData();setInterval(fetchData,5000);
</script></body></html>"""


def main():
    global trader_instance

    parser = argparse.ArgumentParser(description="Quant Engine Live Trader")
    parser.add_argument("--interval", type=int, default=600,
                        help="Rebalance interval in seconds (default: 600 = 10min)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web dashboard port (default: 8080)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest only (no live trading)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override tickers")

    args = parser.parse_args()

    config = SystemConfig()
    if args.tickers:
        config.data.tickers = args.tickers

    if args.backtest:
        # Run backtest using quant engine
        logger.info("Running backtest mode...")
        from backtest.engine import BacktestEngine
        data = DataFeed(config.data)
        data.load()
        models = []
        for name in config.alpha.enabled_models:
            if name == "ts_momentum": models.append(TimeSeriesMomentum(config.alpha))
            elif name == "xs_momentum": models.append(CrossSectionalMomentum(config.alpha))
            elif name == "momentum_vol_break": models.append(MomentumWithVolBreak(config.alpha))
            elif name == "ou_mean_reversion": models.append(OUMeanReversion(config.alpha))
            elif name == "pairs_trading": models.append(PairsTrading(config.alpha))
            elif name == "ml_alpha": models.append(MLAlpha(config.alpha))

        engine = BacktestEngine(config)
        result = engine.run(data, models, initial_capital=1_000_000)
        return

    # Live trading mode
    trader_instance = LiveTrader(config, rebalance_interval=args.interval)

    # Start trading in background thread
    trade_thread = threading.Thread(target=trader_instance.run, daemon=True)
    trade_thread.start()

    # Start web dashboard
    logger.info(f"\n  Live Trading Dashboard: http://localhost:{args.port}\n")
    web_app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
