"""
Unified Trading Dashboard — One-click backtest + live trading.

Press START → runs quant engine backtest → shows equity graph + trade log.
Optionally connects to Alpaca for live paper trading.

Usage:
    python dashboard.py                  # Start dashboard on port 8888
    python dashboard.py --port 9000      # Custom port
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
from backtest.engine import BacktestEngine
from config import SystemConfig
from data.feed import DataFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Dashboard")

app = Flask(__name__)
CORS(app)


# ── JSON encoder for numpy/pandas ────────────────────────────
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
_orig_dumps = _fjp.DefaultJSONProvider.dumps
def _patched_dumps(self, obj, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(obj, **kwargs)
_fjp.DefaultJSONProvider.dumps = _patched_dumps


# ── Global state ──────────────────────────────────────────────
state = {
    "status": "idle",         # idle | loading | running | complete | error
    "progress": 0,
    "progress_msg": "",
    "start_time": None,
    "elapsed": 0,
    "error": None,

    # Live streaming data
    "equity_history": [],     # [{date, equity}]
    "current_day": 0,
    "total_days": 0,
    "initial_capital": 1_000_000,

    # Trade log
    "trade_log": [],          # [{date, n_trades, turnover, total_cost, port_value, details}]

    # Portfolio
    "current_weights": {},
    "model_weights": {},
    "risk_metrics": {},

    # Running metrics
    "metrics": {
        "total_return": 0, "ann_return": 0, "ann_vol": 0,
        "sharpe": 0, "sortino": 0, "max_dd": 0, "current_dd": 0,
        "win_rate": 0, "profit_factor": 0, "calmar": 0,
    },

    # Final results
    "final": None,
}
lock = threading.Lock()


class DashboardCallback:
    """Feeds backtest progress into the global state for the web dashboard."""

    def __init__(self):
        self.peak_equity = 0
        self.initial_capital = 1_000_000
        self.return_history = []

    def on_backtest_start(self, total_days, config_info, initial_capital):
        with lock:
            state["total_days"] = total_days
            state["current_day"] = 0
            state["equity_history"] = []
            state["trade_log"] = []
            state["status"] = "running"
            state["progress_msg"] = "Backtesting..."
            state["initial_capital"] = initial_capital
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital

    def on_day_update(self, date, equity, weights=None, risk_metrics=None,
                      model_weights=None, trade_info=None):
        with lock:
            state["current_day"] += 1
            state["equity_history"].append({
                "date": str(date)[:10],
                "equity": round(equity, 2),
            })

            pct = state["current_day"] / max(state["total_days"], 1) * 100
            state["progress"] = round(pct, 1)

            if weights is not None:
                state["current_weights"] = {
                    k: round(v, 4) for k, v in weights.items() if abs(v) > 0.001
                }
            if risk_metrics is not None:
                state["risk_metrics"] = {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in risk_metrics.items()
                }
            if model_weights is not None:
                state["model_weights"] = {
                    k: round(v, 4) if v == v else 0
                    for k, v in model_weights.items()
                }
            if trade_info is not None:
                state["trade_log"].append({
                    k: (round(v, 4) if isinstance(v, float)
                        else str(v)[:10] if hasattr(v, 'strftime') else v)
                    for k, v in trade_info.items()
                })

            # Running metrics
            eq_list = state["equity_history"]
            n = len(eq_list)
            if n > 1:
                prev_eq = eq_list[-2]["equity"]
                ret = (equity - prev_eq) / prev_eq if prev_eq != 0 else 0
                self.return_history.append(ret)

            self.peak_equity = max(self.peak_equity, equity)
            total_ret = (equity / self.initial_capital) - 1
            dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
            n_years = n / 252

            ann_ret = 0
            ann_vol = 0
            sharpe = 0
            sortino = 0
            win_rate = 0
            profit_factor = 0
            if n_years > 0.05:
                ann_ret = (1 + total_ret) ** (1 / n_years) - 1
            if len(self.return_history) > 20:
                rets = np.array(self.return_history[-252:])
                ann_vol = float(np.std(rets) * np.sqrt(252))
                if ann_vol > 0:
                    sharpe = ann_ret / ann_vol
                down = rets[rets < 0]
                down_vol = float(np.std(down) * np.sqrt(252)) if len(down) > 0 else ann_vol
                sortino = ann_ret / down_vol if down_vol > 0 else 0
                win_rate = float(np.mean(rets > 0))
                gains = float(np.sum(rets[rets > 0]))
                losses = float(np.abs(np.sum(rets[rets < 0])))
                profit_factor = gains / losses if losses > 0 else 0

            prev_max_dd = state["metrics"].get("max_dd", 0)
            state["metrics"] = {
                "total_return": round(total_ret, 6),
                "ann_return": round(ann_ret, 6),
                "ann_vol": round(ann_vol, 6),
                "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4),
                "max_dd": round(min(dd, prev_max_dd), 6),
                "current_dd": round(dd, 6),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 4),
                "calmar": round(ann_ret / abs(min(dd, prev_max_dd)) if min(dd, prev_max_dd) < -0.001 else 0, 4),
            }

    def on_backtest_complete(self):
        with lock:
            state["progress"] = 100
            state["progress_msg"] = "Complete"

    def show_final_report(self, result):
        pass  # We handle this in the run thread


def _run_backtest(config_overrides: dict):
    """Run the full backtest pipeline in a background thread."""
    try:
        with lock:
            state["status"] = "loading"
            state["progress"] = 0
            state["progress_msg"] = "Initializing..."
            state["error"] = None
            state["final"] = None
            state["equity_history"] = []
            state["trade_log"] = []
            state["start_time"] = time.time()

        config = SystemConfig()

        # Apply overrides
        if "tickers" in config_overrides and config_overrides["tickers"]:
            config.data.tickers = config_overrides["tickers"]
        if "years" in config_overrides:
            config.data.years = int(config_overrides["years"])
        capital = float(config_overrides.get("capital", 1_000_000))

        with lock:
            state["progress_msg"] = "Downloading market data..."
            state["progress"] = 5

        feed = DataFeed(config.data)
        feed.fetch()
        logger.info(f"Data loaded: {feed.prices.shape}")

        with lock:
            state["progress_msg"] = "Building alpha models..."
            state["progress"] = 15

        # Build models
        alpha_cfg = config.alpha
        models = []
        for name in alpha_cfg.enabled_models:
            if name == "ts_momentum": models.append(TimeSeriesMomentum(alpha_cfg))
            elif name == "xs_momentum": models.append(CrossSectionalMomentum(alpha_cfg))
            elif name == "momentum_vol_break": models.append(MomentumWithVolBreak(alpha_cfg))
            elif name == "ou_mean_reversion": models.append(OUMeanReversion(alpha_cfg))
            elif name == "pairs_trading": models.append(PairsTrading(alpha_cfg))
            elif name == "ml_alpha": models.append(MLAlpha(alpha_cfg))
        if not models:
            models = [
                TimeSeriesMomentum(alpha_cfg), CrossSectionalMomentum(alpha_cfg),
                MomentumWithVolBreak(alpha_cfg), OUMeanReversion(alpha_cfg),
                PairsTrading(alpha_cfg), MLAlpha(alpha_cfg),
            ]

        with lock:
            state["progress_msg"] = "Running backtest..."
            state["progress"] = 20

        callback = DashboardCallback()
        callback.on_backtest_start(
            total_days=len(feed.prices),
            config_info={},
            initial_capital=capital,
        )

        engine = BacktestEngine(config)
        result = engine.run(feed, models, initial_capital=capital, dashboard=callback)

        # Package final results
        perf = result.performance_metrics
        ex = result.execution_summary

        # Build detailed trade log from execution engine
        detailed_trades = []
        for t in result.trade_log:
            detailed_trades.append({
                "date": str(t.get("date", ""))[:10],
                "n_trades": t.get("n_trades", 0),
                "turnover": round(t.get("turnover", 0), 4),
                "cost": round(t.get("total_cost", 0), 2),
                "port_value": round(t.get("port_value", 0), 2),
            })

        # Equity curve for final results
        eq_dates = [str(d)[:10] for d in result.equity_curve.index]
        eq_vals = [round(v, 2) for v in result.equity_curve.values]

        # Drawdown series
        cumret = result.equity_curve / result.equity_curve.iloc[0]
        running_max = cumret.cummax()
        drawdown = ((cumret - running_max) / running_max).values.tolist()

        # Rolling Sharpe
        rolling_sharpe = []
        if len(result.returns) > 63:
            rs = (result.returns.rolling(63).mean() / result.returns.rolling(63).std() * np.sqrt(252))
            rolling_sharpe = [round(v, 4) if not np.isnan(v) else 0 for v in rs.values]

        # Weights over time
        weights_hist = {}
        if not result.weights_history.empty:
            weights_hist["_dates"] = [str(d)[:10] for d in result.weights_history.index]
            for col in result.weights_history.columns:
                weights_hist[col] = [
                    round(v, 4) if not np.isnan(v) else 0
                    for v in result.weights_history[col].values
                ]

        final_result = {
            "performance": {k: round(v, 6) if isinstance(v, float) else v for k, v in perf.items()},
            "execution": {k: round(v, 4) if isinstance(v, float) else v for k, v in ex.items()},
            "equity_curve": {"dates": eq_dates, "values": eq_vals},
            "drawdown": [round(v, 6) for v in drawdown],
            "rolling_sharpe": rolling_sharpe,
            "weights_history": weights_hist,
            "detailed_trades": detailed_trades,
        }

        with lock:
            state["status"] = "complete"
            state["progress"] = 100
            state["progress_msg"] = "Complete"
            state["final"] = final_result
            state["elapsed"] = round(time.time() - state["start_time"], 1)
            state["trade_log"] = detailed_trades

        logger.info(f"Backtest complete in {state['elapsed']}s")

    except Exception as e:
        logger.exception("Backtest failed")
        with lock:
            state["status"] = "error"
            state["error"] = str(e)
            state["progress_msg"] = f"Error: {e}"


# ── API Routes ────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    with lock:
        eq = state["equity_history"]
        # Downsample for live streaming (max 1000 points)
        if len(eq) > 1000:
            step = len(eq) // 1000
            eq = eq[::step]

        return jsonify({
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
            "metrics": state["metrics"],
            "trade_count": len(state["trade_log"]),
        })


@app.route("/api/trades")
def api_trades():
    with lock:
        page = int(request.args.get("page", 0))
        size = int(request.args.get("size", 50))
        trades = state["trade_log"]
        # Return newest first
        start = max(0, len(trades) - (page + 1) * size)
        end = len(trades) - page * size
        return jsonify({
            "trades": list(reversed(trades[start:end])),
            "total": len(trades),
            "page": page,
            "pages": max(1, (len(trades) + size - 1) // size),
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
        if state["status"] in ("loading", "running"):
            return jsonify({"error": "Already running"}), 409

    config_overrides = request.json or {}
    thread = threading.Thread(target=_run_backtest, args=(config_overrides,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with lock:
        state["status"] = "idle"
        state["progress"] = 0
        state["progress_msg"] = ""
        state["error"] = None
        state["final"] = None
        state["equity_history"] = []
        state["trade_log"] = []
        state["current_weights"] = {}
        state["model_weights"] = {}
        state["risk_metrics"] = {}
        state["metrics"] = {
            "total_return": 0, "ann_return": 0, "ann_vol": 0,
            "sharpe": 0, "sortino": 0, "max_dd": 0, "current_dd": 0,
            "win_rate": 0, "profit_factor": 0, "calmar": 0,
        }
    return jsonify({"status": "reset"})


@app.route("/")
def index():
    return Response(DASHBOARD_HTML, mimetype="text/html")


# ── Dashboard HTML ────────────────────────────────────────────
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

/* Header */
.header{background:linear-gradient(180deg,#0f1728,var(--bg1));border-bottom:1px solid var(--border);padding:12px 24px;position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.brand{font-size:18px;font-weight:800;color:var(--gold);letter-spacing:-.5px}
.brand small{font-size:10px;color:var(--text3);font-weight:400;letter-spacing:1px;display:block;margin-top:-2px}
.hdr-right{display:flex;align-items:center;gap:10px}
.badge{padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.badge-idle{background:var(--blue-bg);color:var(--blue)}
.badge-running{background:var(--green-bg);color:var(--green);animation:pulse 1.5s infinite}
.badge-complete{background:var(--purple-bg);color:var(--purple)}
.badge-error{background:var(--red-bg);color:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

.btn{padding:8px 20px;border-radius:6px;border:1px solid var(--border);background:var(--bg3);color:var(--text);font-family:inherit;font-size:12px;cursor:pointer;transition:.15s;font-weight:700}
.btn:hover{background:var(--bg4);border-color:var(--border-light)}
.btn-start{background:var(--green-dim);border-color:var(--green);color:var(--green);font-size:14px;padding:10px 32px;letter-spacing:1px}
.btn-start:hover{background:#047857}
.btn-start:disabled{opacity:.3;cursor:not-allowed}
.btn-reset{background:var(--bg3);border-color:var(--border);color:var(--text2);font-size:11px}
.elapsed{font-size:11px;color:var(--text3)}

/* Main layout */
.main{max-width:1600px;margin:0 auto;padding:14px}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}

/* Hero */
.hero{grid-column:1/-1;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px}
.port-val{font-size:48px;font-weight:800;line-height:1}
.port-chg{font-size:15px;margin-top:4px}
.port-stats{display:flex;gap:24px;flex-wrap:wrap}
.stat{display:flex;flex-direction:column;gap:1px}
.stat-l{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px}
.stat-v{font-size:14px;font-weight:700}

/* Progress */
.progress-wrap{grid-column:1/-1}
.prog-bar{height:20px;background:var(--bg1);border-radius:10px;overflow:hidden;border:1px solid var(--border)}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--green-dim),var(--green));border-radius:10px;transition:width .4s}
.prog-label{font-size:10px;color:var(--text3);margin-top:4px;text-align:center}

/* Cards */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.card-h{padding:8px 14px;border-bottom:1px solid var(--border);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--text3);display:flex;align-items:center;justify-content:space-between}
.card-b{padding:14px}

/* Charts */
.chart-area{grid-column:1/3;min-height:320px}
.chart-area .card-b{padding:8px;height:300px}
.chart-dd{grid-column:1/3;min-height:180px}
.chart-dd .card-b{padding:8px;height:160px}

/* Metrics */
.m-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(30,45,69,.3);font-size:12px}
.m-row:last-child{border-bottom:none}
.m-label{color:var(--text2)}
.m-val{font-weight:700}

/* Model weights bars */
.w-bar-wrap{margin:4px 0}
.w-bar-label{display:flex;justify-content:space-between;font-size:10px;margin-bottom:2px}
.w-bar-bg{height:12px;background:var(--bg0);border-radius:6px;overflow:hidden}
.w-bar-fill{height:100%;border-radius:6px;transition:width .4s}

/* Trade log */
.trades-card{grid-column:1/-1}
.trades-table{width:100%;border-collapse:collapse;font-size:11px}
.trades-table th{text-align:left;padding:6px 10px;color:var(--text3);border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0;background:var(--bg2)}
.trades-table td{padding:6px 10px;border-bottom:1px solid rgba(30,45,69,.3)}
.trades-table tr:hover td{background:var(--bg3)}
.trades-scroll{max-height:400px;overflow-y:auto}
.page-controls{display:flex;justify-content:center;gap:8px;padding:8px;font-size:11px}
.page-btn{padding:4px 12px;border-radius:4px;border:1px solid var(--border);background:var(--bg3);color:var(--text2);cursor:pointer;font-family:inherit;font-size:10px}
.page-btn:hover{background:var(--bg4)}
.page-btn.active{background:var(--gold);color:var(--bg0);border-color:var(--gold)}

/* Tabs */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:12px}
.tab{padding:8px 18px;font-size:11px;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;transition:.15s;background:none;border-top:none;border-left:none;border-right:none;font-family:inherit}
.tab:hover{color:var(--text2)}
.tab.active{color:var(--gold);border-bottom-color:var(--gold)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* Utilities */
.pos{color:var(--green)}.neg{color:var(--red)}
.hidden{display:none}
.empty{color:var(--text3);font-size:12px;text-align:center;padding:40px 0}

/* Responsive */
@media(max-width:900px){.grid{grid-template-columns:1fr}.chart-area,.chart-dd,.trades-card{grid-column:1/-1}}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="brand">Quant Trading Dashboard<small>Powered by Quant Engine</small></div>
  <div class="hdr-right">
    <span id="status-badge" class="badge badge-idle">IDLE</span>
    <span id="elapsed" class="elapsed"></span>
    <button id="btn-start" class="btn btn-start" onclick="startBacktest()">START</button>
    <button class="btn btn-reset" onclick="resetDashboard()">RESET</button>
  </div>
</div>

<!-- Tabs -->
<div class="main">
<div class="tabs">
  <button class="tab active" onclick="switchTab('dashboard',this)">Dashboard</button>
  <button class="tab" onclick="switchTab('trades',this)">Trade Log</button>
  <button class="tab" onclick="switchTab('weights',this)">Weights & Risk</button>
</div>

<!-- ═══ Dashboard Tab ═══ -->
<div id="tab-dashboard" class="tab-panel active">
<div class="grid">

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
          <div class="stat"><span class="stat-l">Max Drawdown</span><span class="stat-v" id="s-max-dd">—</span></div>
          <div class="stat"><span class="stat-l">Win Rate</span><span class="stat-v" id="s-win-rate">—</span></div>
          <div class="stat"><span class="stat-l">Profit Factor</span><span class="stat-v" id="s-pf">—</span></div>
          <div class="stat"><span class="stat-l">Calmar</span><span class="stat-v" id="s-calmar">—</span></div>
          <div class="stat"><span class="stat-l">Trades</span><span class="stat-v" id="s-trades">0</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Progress -->
  <div class="progress-wrap" id="progress-wrap">
    <div class="prog-bar"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="prog-label" id="prog-label">Press START to begin</div>
  </div>

  <!-- Equity Chart -->
  <div class="chart-area card">
    <div class="card-h"><span>Equity Curve</span><span id="chart-info"></span></div>
    <div class="card-b"><canvas id="equityChart"></canvas></div>
  </div>

  <!-- Metrics -->
  <div class="card metrics-card">
    <div class="card-h">Performance Metrics</div>
    <div class="card-b" id="metrics-body">
      <div class="empty">Run a backtest to see metrics</div>
    </div>
  </div>

  <!-- Drawdown Chart -->
  <div class="chart-dd card">
    <div class="card-h">Drawdown</div>
    <div class="card-b"><canvas id="ddChart"></canvas></div>
  </div>

  <!-- Model Weights -->
  <div class="card">
    <div class="card-h">Model Weights</div>
    <div class="card-b" id="model-weights-body">
      <div class="empty">Waiting for data...</div>
    </div>
  </div>

</div>
</div>

<!-- ═══ Trade Log Tab ═══ -->
<div id="tab-trades" class="tab-panel">
<div class="card trades-card">
  <div class="card-h"><span>Trade Log</span><span id="trade-count">0 trades</span></div>
  <div class="trades-scroll" id="trades-scroll">
    <table class="trades-table">
      <thead><tr><th>#</th><th>Date</th><th>Trades</th><th>Turnover</th><th>Cost</th><th>Portfolio Value</th></tr></thead>
      <tbody id="trades-body"><tr><td colspan="6" class="empty">No trades yet</td></tr></tbody>
    </table>
  </div>
  <div class="page-controls" id="page-controls"></div>
</div>
</div>

<!-- ═══ Weights & Risk Tab ═══ -->
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

</div><!-- /main -->

<script>
// ── State ──
let equityChart = null, ddChart = null;
let pollInterval = null;
let currentPage = 0;
const PAGE_SIZE = 50;

// ── Tab switching ──
function switchTab(name, el) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
  if (name === 'trades') loadTrades(0);
}

// ── Formatting ──
function $(s){return document.getElementById(s)}
function money(v){return '$' + Number(v).toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:0})}
function pct(v){return (v*100).toFixed(2)+'%'}
function pctCls(v){return v>=0?'pos':'neg'}
function num(v,d=2){return Number(v).toFixed(d)}

// ── Charts ──
function initCharts() {
  const eq = $('equityChart').getContext('2d');
  equityChart = new Chart(eq, {
    type:'line',
    data:{labels:[],datasets:[{
      label:'Equity',data:[],
      borderColor:'#10b981',backgroundColor:'rgba(16,185,129,.08)',
      borderWidth:2,pointRadius:0,fill:true,tension:.1
    }]},
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:{duration:0},
      scales:{
        x:{display:true,ticks:{maxTicksLimit:10,color:'#5a6a84',font:{size:9}},grid:{color:'rgba(30,45,69,.3)'}},
        y:{ticks:{color:'#94a3b8',font:{size:10},callback:v=>'$'+(v/1e6).toFixed(2)+'M'},grid:{color:'rgba(30,45,69,.3)'}}
      },
      plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false}}
    }
  });
  const dd = $('ddChart').getContext('2d');
  ddChart = new Chart(dd, {
    type:'line',
    data:{labels:[],datasets:[{
      label:'Drawdown',data:[],
      borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,.15)',
      borderWidth:1.5,pointRadius:0,fill:true,tension:.1
    }]},
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:{duration:0},
      scales:{
        x:{display:true,ticks:{maxTicksLimit:8,color:'#5a6a84',font:{size:9}},grid:{color:'rgba(30,45,69,.3)'}},
        y:{ticks:{color:'#94a3b8',font:{size:10},callback:v=>pct(v)},grid:{color:'rgba(30,45,69,.3)'}}
      },
      plugins:{legend:{display:false}}
    }
  });
}

// ── Start backtest ──
async function startBacktest() {
  $('btn-start').disabled = true;
  try {
    await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    startPolling();
  } catch(e) {
    alert('Failed to start: ' + e);
    $('btn-start').disabled = false;
  }
}

// ── Reset ──
async function resetDashboard() {
  await fetch('/api/reset', {method:'POST'});
  stopPolling();
  $('btn-start').disabled = false;
  $('status-badge').className = 'badge badge-idle';
  $('status-badge').textContent = 'IDLE';
  $('port-val').textContent = '$1,000,000';
  $('port-chg').textContent = '+0.00%';
  $('port-chg').className = 'port-chg';
  $('prog-fill').style.width = '0%';
  $('prog-label').textContent = 'Press START to begin';
  $('elapsed').textContent = '';
  ['s-ann-ret','s-sharpe','s-sortino','s-max-dd','s-win-rate','s-pf','s-calmar'].forEach(id => $(id).textContent = '—');
  $('s-trades').textContent = '0';
  $('metrics-body').innerHTML = '<div class="empty">Run a backtest to see metrics</div>';
  $('model-weights-body').innerHTML = '<div class="empty">Waiting for data...</div>';
  $('trades-body').innerHTML = '<tr><td colspan="6" class="empty">No trades yet</td></tr>';
  $('trade-count').textContent = '0 trades';
  if(equityChart){equityChart.data.labels=[];equityChart.data.datasets[0].data=[];equityChart.update()}
  if(ddChart){ddChart.data.labels=[];ddChart.data.datasets[0].data=[];ddChart.update()}
}

// ── Polling ──
function startPolling() { if(!pollInterval) pollInterval = setInterval(pollStatus, 800); }
function stopPolling() { if(pollInterval){clearInterval(pollInterval);pollInterval=null} }

async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    updateUI(d);
    if (d.status === 'complete' || d.status === 'error') {
      stopPolling();
      $('btn-start').disabled = false;
      if (d.status === 'complete') loadFinalResults();
    }
  } catch(e) { console.error(e); }
}

function updateUI(d) {
  // Badge
  const badgeMap = {idle:'badge-idle',loading:'badge-running',running:'badge-running',complete:'badge-complete',error:'badge-error'};
  $('status-badge').className = 'badge ' + (badgeMap[d.status]||'badge-idle');
  $('status-badge').textContent = d.status.toUpperCase();

  // Elapsed
  if(d.elapsed > 0) $('elapsed').textContent = num(d.elapsed,1) + 's';

  // Progress
  $('prog-fill').style.width = d.progress + '%';
  $('prog-label').textContent = d.progress_msg || (d.status==='running' ? `Day ${d.current_day}/${d.total_days}` : '');

  // Equity
  const eq = d.equity_history;
  if(eq && eq.length > 0) {
    const latest = eq[eq.length-1];
    const val = latest.equity;
    $('port-val').textContent = money(val);
    const initCap = d.metrics?.total_return !== undefined ? val / (1 + d.metrics.total_return) : 1000000;
    const totalRet = d.metrics?.total_return || 0;
    $('port-chg').textContent = (totalRet>=0?'+':'') + pct(totalRet);
    $('port-chg').className = 'port-chg ' + pctCls(totalRet);

    // Update chart
    const labels = eq.map(e => e.date);
    const vals = eq.map(e => e.equity);
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = vals;
    equityChart.update();

    // Drawdown
    let peak = vals[0];
    const dds = vals.map(v => { peak = Math.max(peak, v); return (v - peak) / peak; });
    ddChart.data.labels = labels;
    ddChart.data.datasets[0].data = dds;
    ddChart.update();

    $('chart-info').textContent = `${eq.length} days`;
  }

  // Metrics
  const m = d.metrics;
  if(m) {
    $('s-ann-ret').textContent = pct(m.ann_return);
    $('s-ann-ret').className = 'stat-v ' + pctCls(m.ann_return);
    $('s-sharpe').textContent = num(m.sharpe);
    $('s-sharpe').className = 'stat-v ' + pctCls(m.sharpe);
    $('s-sortino').textContent = num(m.sortino);
    $('s-max-dd').textContent = pct(m.max_dd);
    $('s-max-dd').className = 'stat-v neg';
    $('s-win-rate').textContent = pct(m.win_rate);
    $('s-pf').textContent = num(m.profit_factor);
    $('s-calmar').textContent = num(m.calmar);
    $('s-trades').textContent = d.trade_count || 0;

    // Metrics card
    $('metrics-body').innerHTML = [
      ['Total Return', pct(m.total_return), pctCls(m.total_return)],
      ['Ann. Return', pct(m.ann_return), pctCls(m.ann_return)],
      ['Ann. Volatility', pct(m.ann_vol), ''],
      ['Sharpe Ratio', num(m.sharpe), pctCls(m.sharpe)],
      ['Sortino Ratio', num(m.sortino), pctCls(m.sortino)],
      ['Max Drawdown', pct(m.max_dd), 'neg'],
      ['Current Drawdown', pct(m.current_dd), 'neg'],
      ['Win Rate', pct(m.win_rate), m.win_rate>0.5?'pos':''],
      ['Profit Factor', num(m.profit_factor), m.profit_factor>1?'pos':'neg'],
      ['Calmar Ratio', num(m.calmar), pctCls(m.calmar)],
    ].map(([l,v,c]) => `<div class="m-row"><span class="m-label">${l}</span><span class="m-val ${c}">${v}</span></div>`).join('');
  }

  // Model weights
  const mw = d.model_weights;
  if(mw && Object.keys(mw).length > 0) {
    const sorted = Object.entries(mw).sort((a,b) => b[1]-a[1]);
    const maxW = Math.max(...sorted.map(([,v])=>v), 0.01);
    const colors = ['#10b981','#3b82f6','#a78bfa','#22d3ee','#fb923c','#ec4899'];
    $('model-weights-body').innerHTML = sorted.map(([name,w], i) => {
      const pctW = (w*100).toFixed(1);
      const barW = (w/maxW*100).toFixed(0);
      return `<div class="w-bar-wrap">
        <div class="w-bar-label"><span>${name}</span><span>${pctW}%</span></div>
        <div class="w-bar-bg"><div class="w-bar-fill" style="width:${barW}%;background:${colors[i%6]}"></div></div>
      </div>`;
    }).join('');
  }

  // Weights tab
  const cw = d.current_weights;
  if(cw && Object.keys(cw).length > 0) {
    const sorted = Object.entries(cw).sort((a,b) => Math.abs(b[1])-Math.abs(a[1]));
    $('weights-body').innerHTML = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px">' +
      sorted.map(([ticker,w]) => {
        const cls = w>0?'pos':'neg';
        return `<div style="background:var(--bg1);padding:8px;border-radius:4px;border:1px solid var(--border)">
          <div style="font-weight:800;font-size:12px">${ticker}</div>
          <div class="${cls}" style="font-size:14px;font-weight:700">${pct(w)}</div>
        </div>`;
      }).join('') + '</div>';
  }

  // Risk tab
  const rm = d.risk_metrics;
  if(rm && Object.keys(rm).length > 0) {
    $('risk-body').innerHTML = Object.entries(rm).map(([k,v]) => {
      const val = typeof v === 'number' ? (Math.abs(v)<1 ? pct(v) : num(v,4)) : String(v);
      return `<div class="m-row"><span class="m-label">${k}</span><span class="m-val">${val}</span></div>`;
    }).join('');
  }
}

// ── Load final results ──
async function loadFinalResults() {
  try {
    const r = await fetch('/api/result');
    if(!r.ok) return;
    const d = await r.json();

    // Update equity chart with full resolution
    if(d.equity_curve) {
      equityChart.data.labels = d.equity_curve.dates;
      equityChart.data.datasets[0].data = d.equity_curve.values;
      equityChart.update();
    }

    // Drawdown
    if(d.drawdown) {
      ddChart.data.labels = d.equity_curve?.dates || [];
      ddChart.data.datasets[0].data = d.drawdown;
      ddChart.update();
    }

    // Final metrics
    const p = d.performance;
    if(p) {
      $('port-val').textContent = money(p.final_equity);
      $('port-chg').textContent = (p.total_return>=0?'+':'') + pct(p.total_return);
      $('port-chg').className = 'port-chg ' + pctCls(p.total_return);
      $('s-ann-ret').textContent = pct(p.annualized_return);
      $('s-sharpe').textContent = num(p.sharpe_ratio);
      $('s-sortino').textContent = num(p.sortino_ratio);
      $('s-max-dd').textContent = pct(p.max_drawdown);
      $('s-win-rate').textContent = pct(p.win_rate);
      $('s-pf').textContent = num(p.profit_factor);
      $('s-calmar').textContent = num(p.calmar_ratio);
      $('s-trades').textContent = d.detailed_trades?.length || 0;

      $('metrics-body').innerHTML = [
        ['Total Return', pct(p.total_return), pctCls(p.total_return)],
        ['Annualized Return', pct(p.annualized_return), pctCls(p.annualized_return)],
        ['Annualized Vol', pct(p.annualized_volatility), ''],
        ['Sharpe Ratio', num(p.sharpe_ratio), pctCls(p.sharpe_ratio)],
        ['Sortino Ratio', num(p.sortino_ratio), pctCls(p.sortino_ratio)],
        ['Max Drawdown', pct(p.max_drawdown), 'neg'],
        ['Calmar Ratio', num(p.calmar_ratio), pctCls(p.calmar_ratio)],
        ['Win Rate', pct(p.win_rate), p.win_rate>0.5?'pos':''],
        ['Profit Factor', num(p.profit_factor), p.profit_factor>1?'pos':'neg'],
        ['Skewness', num(p.skewness), pctCls(-p.skewness)],
        ['Kurtosis', num(p.kurtosis), ''],
        ['Trading Days', p.n_trading_days, ''],
      ].map(([l,v,c]) => `<div class="m-row"><span class="m-label">${l}</span><span class="m-val ${c}">${v}</span></div>`).join('');

      // Execution stats
      if(d.execution) {
        $('metrics-body').innerHTML += '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">' +
          [
            ['Total Costs', money(d.execution.total_costs), 'neg'],
            ['Total Turnover', num(d.execution.total_turnover)+'x', ''],
            ['Rebalances', d.execution.n_rebalances, ''],
            ['Avg Cost/Rebal', money(d.execution.avg_cost_per_rebalance), ''],
          ].map(([l,v,c]) => `<div class="m-row"><span class="m-label">${l}</span><span class="m-val ${c||''}">${v}</span></div>`).join('') +
          '</div>';
      }
    }

    // Load trades
    loadTrades(0);

  } catch(e) { console.error(e); }
}

// ── Trade log ──
async function loadTrades(page) {
  currentPage = page;
  try {
    const r = await fetch(`/api/trades?page=${page}&size=${PAGE_SIZE}`);
    const d = await r.json();
    $('trade-count').textContent = `${d.total} trades`;

    if(!d.trades || d.trades.length === 0) {
      $('trades-body').innerHTML = '<tr><td colspan="6" class="empty">No trades yet</td></tr>';
      $('page-controls').innerHTML = '';
      return;
    }

    $('trades-body').innerHTML = d.trades.map((t, i) => {
      const idx = d.total - page * PAGE_SIZE - i;
      const costCls = t.cost > 500 ? 'neg' : '';
      return `<tr>
        <td style="color:var(--text3)">${idx}</td>
        <td>${t.date}</td>
        <td>${t.n_trades}</td>
        <td>${num(t.turnover * 100, 1)}%</td>
        <td class="${costCls}">${money(t.cost)}</td>
        <td>${money(t.port_value)}</td>
      </tr>`;
    }).join('');

    // Pagination
    if(d.pages > 1) {
      let btns = '';
      for(let p = 0; p < Math.min(d.pages, 10); p++) {
        btns += `<button class="page-btn ${p===page?'active':''}" onclick="loadTrades(${p})">${p+1}</button>`;
      }
      if(d.pages > 10) btns += `<span style="color:var(--text3);padding:4px">...${d.pages} pages</span>`;
      $('page-controls').innerHTML = btns;
    } else {
      $('page-controls').innerHTML = '';
    }
  } catch(e) { console.error(e); }
}

// ── Init ──
initCharts();
// Check if there's already a running backtest
(async()=>{
  try{
    const r = await fetch('/api/status');
    const d = await r.json();
    if(d.status==='running'||d.status==='loading'){
      $('btn-start').disabled=true;
      startPolling();
    } else if(d.status==='complete'){
      updateUI(d);
      loadFinalResults();
    }
  }catch(e){}
})();
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
    print(f"  Open: http://{args.host}:{args.port}")
    print(f"  Press START to run the backtest\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
