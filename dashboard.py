"""
Auto-Trader Dashboard — Local web UI for backtest visualization.
Run: python dashboard.py
Then open http://localhost:8050
"""
import os
import sys
import json
import glob
from datetime import datetime

from flask import Flask, render_template_string

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load_latest_backtest():
    """Load the most recent backtest JSON."""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "backtest_*.json")))
    if not files:
        return None
    with open(files[-1], "r") as f:
        return json.load(f)


def run_backtest_with_details():
    """Run a fresh backtest and capture detailed trade/equity data."""
    from config import INITIAL_CAPITAL
    from data.fetcher import fetch_ohlcv, build_universe
    from features.engineering import (
        build_features_universe, build_cross_sectional_features,
        prepare_ml_dataset,
    )
    from backtest.walk_forward import WalkForwardBacktester

    print("[Dashboard] Running backtest...")
    data = fetch_ohlcv()
    universe = build_universe(data)
    features = build_features_universe(data)
    features = build_cross_sectional_features(features, universe)
    dataset = prepare_ml_dataset(features, data, forward_days=7)

    bt = WalkForwardBacktester()
    results = bt.run(dataset, features, data)

    # Extract detailed data for dashboard
    trades = []
    for t in bt.trades:
        trades.append({
            "symbol": t.symbol,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_date": str(t.entry_date)[:10] if t.entry_date else "",
            "exit_date": str(t.exit_date)[:10] if t.exit_date else "",
            "entry_price": round(t.entry_price, 2) if t.entry_price else 0,
            "exit_price": round(t.exit_price, 2) if t.exit_price else 0,
            "size": round(t.size, 2),
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct * 100, 2),
            "signal_strength": round(t.signal_strength, 3),
            "regime": {1: "Trending", -1: "Mean-Rev", 0: "Neutral"}.get(t.regime, "?"),
            "exit_reason": t.exit_reason,
            "won": t.pnl > 0,
        })

    equity = [{"date": str(e["date"])[:10], "capital": round(e["capital"], 2),
               "fold": e["fold"], "positions": e["open_positions"]}
              for e in bt.equity_curve]

    return results, trades, equity, bt.fold_results


# Cache results at startup
print("=" * 50)
print("  Auto-Trader Dashboard")
print("  Running backtest, please wait...")
print("=" * 50)
_results, _trades, _equity, _folds = run_backtest_with_details()


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auto-Trader Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e1a; color: #c8d0e0; }
.header { background: linear-gradient(135deg, #0f1629 0%, #1a1f3a 100%); padding: 20px 30px;
  border-bottom: 1px solid #1e2744; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.5rem; color: #e2e8f0; }
.header h1 span { color: #22d3ee; }
.header .ts { color: #64748b; font-size: 0.8rem; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }
.stat-card { background: #111827; border: 1px solid #1e2744; border-radius: 10px; padding: 16px;
  text-align: center; transition: border-color 0.2s; }
.stat-card:hover { border-color: #334155; }
.stat-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.stat-value { font-size: 1.6rem; font-weight: 700; }
.stat-sub { font-size: 0.72rem; color: #64748b; margin-top: 4px; }
.green { color: #22c55e; } .red { color: #ef4444; } .blue { color: #3b82f6; }
.cyan { color: #22d3ee; } .orange { color: #f97316; } .purple { color: #a78bfa; }
.chart-container { background: #111827; border: 1px solid #1e2744; border-radius: 10px;
  padding: 20px; margin-bottom: 24px; }
.chart-container h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 14px; }
.chart-wrap { height: 340px; position: relative; }
.section { background: #111827; border: 1px solid #1e2744; border-radius: 10px;
  padding: 20px; margin-bottom: 24px; }
.section h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 14px; }
.tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.tab { padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.8rem;
  background: #1e293b; color: #94a3b8; border: 1px solid #334155; transition: all 0.2s; }
.tab.active { background: #22d3ee; color: #0a0e1a; border-color: #22d3ee; font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
th { background: #1e293b; color: #94a3b8; padding: 8px 10px; text-align: left;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.68rem;
  position: sticky; top: 0; }
td { padding: 7px 10px; border-bottom: 1px solid #1e2744; }
tr:hover { background: #1a2235; }
.badge { padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 600; }
.badge-long { background: #064e3b; color: #34d399; }
.badge-short { background: #7f1d1d; color: #f87171; }
.badge-win { background: #064e3b; color: #22c55e; }
.badge-loss { background: #7f1d1d; color: #ef4444; }
.badge-trending { background: #1e3a5f; color: #60a5fa; }
.badge-mr { background: #3b1f5e; color: #c084fc; }
.badge-neutral { background: #374151; color: #9ca3af; }
.scroll-table { max-height: 500px; overflow-y: auto; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
.fold-card { background: #0f1629; border: 1px solid #1e2744; border-radius: 8px; padding: 12px;
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.fold-card .fold-id { font-weight: 700; color: #22d3ee; width: 60px; }
.fold-card .fold-stats { display: flex; gap: 16px; font-size: 0.78rem; }
.mini-bar { height: 6px; border-radius: 3px; margin-top: 4px; }
</style>
</head>
<body>
<div class="header">
  <h1><span>&#9883;</span> Auto-Trader <span>Dashboard</span></h1>
  <div class="ts">Backtest completed {{ timestamp }}</div>
</div>
<div class="container">

<!-- Stats Grid -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Total Return</div>
    <div class="stat-value {{ 'green' if total_return >= 0 else 'red' }}">{{ '%+.1f'|format(total_return) }}%</div>
    <div class="stat-sub">${{ '{:,.0f}'.format(initial) }} &rarr; ${{ '{:,.0f}'.format(final) }}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Sharpe Ratio</div>
    <div class="stat-value {{ 'green' if sharpe >= 1.0 else 'orange' if sharpe >= 0.5 else 'red' }}">{{ '%.3f'|format(sharpe) }}</div>
    <div class="stat-sub">Risk-adjusted</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Max Drawdown</div>
    <div class="stat-value red">{{ '%.1f'|format(max_dd) }}%</div>
    <div class="stat-sub">Peak to trough</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Win Rate</div>
    <div class="stat-value {{ 'green' if win_rate >= 50 else 'orange' }}">{{ '%.1f'|format(win_rate) }}%</div>
    <div class="stat-sub">{{ n_wins }}W / {{ n_losses }}L</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Profit Factor</div>
    <div class="stat-value {{ 'green' if pf >= 1.2 else 'orange' }}">{{ '%.2f'|format(pf) }}</div>
    <div class="stat-sub">Gross win/loss</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Total Trades</div>
    <div class="stat-value cyan">{{ n_trades }}</div>
    <div class="stat-sub">{{ n_folds }} folds</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Win</div>
    <div class="stat-value green">${{ '%.0f'|format(avg_win) }}</div>
    <div class="stat-sub">Per winning trade</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Loss</div>
    <div class="stat-value red">${{ '%.0f'|format(avg_loss) }}</div>
    <div class="stat-sub">Per losing trade</div>
  </div>
</div>

<!-- Equity Curve -->
<div class="chart-container">
  <h2>Equity Curve &mdash; Walk-Forward Backtest</h2>
  <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
</div>

<!-- Two Column: Drawdown + P&L Distribution -->
<div class="two-col">
  <div class="chart-container">
    <h2>Drawdown</h2>
    <div class="chart-wrap"><canvas id="ddChart"></canvas></div>
  </div>
  <div class="chart-container">
    <h2>Trade P&L Distribution</h2>
    <div class="chart-wrap"><canvas id="pnlHist"></canvas></div>
  </div>
</div>

<!-- Fold Results -->
<div class="section">
  <h2>Walk-Forward Fold Results</h2>
  {% for f in folds %}
  <div class="fold-card">
    <div class="fold-id">Fold {{ f.fold }}</div>
    <div class="fold-stats">
      <span>{{ f.n_trades }} trades</span>
      <span class="{{ 'green' if f.win_rate >= 50 else 'orange' }}">{{ '%.0f'|format(f.win_rate) }}% win</span>
      <span class="{{ 'green' if f.roi_pct >= 0 else 'red' }}">{{ '%+.1f'|format(f.roi_pct) }}% ROI</span>
      <span>${{ '{:,.0f}'.format(f.starting_capital) }} &rarr; ${{ '{:,.0f}'.format(f.ending_capital) }}</span>
    </div>
  </div>
  {% endfor %}
</div>

<!-- Trade Log -->
<div class="section">
  <h2>Trade Log ({{ n_trades }} trades)</h2>
  <div class="tabs">
    <div class="tab active" onclick="filterTrades('all',this)">All</div>
    <div class="tab" onclick="filterTrades('win',this)">Winners</div>
    <div class="tab" onclick="filterTrades('loss',this)">Losers</div>
    <div class="tab" onclick="filterTrades('long',this)">Longs</div>
    <div class="tab" onclick="filterTrades('short',this)">Shorts</div>
  </div>
  <div class="scroll-table">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
          <th>Entry $</th><th>Exit $</th><th>Size</th><th>P&L</th><th>P&L%</th>
          <th>Signal</th><th>Regime</th><th>Exit Reason</th>
        </tr>
      </thead>
      <tbody id="trade-body">
        {% for t in trades %}
        <tr class="trade-row" data-won="{{ 'win' if t.won else 'loss' }}"
            data-dir="{{ t.direction|lower }}">
          <td>{{ loop.index }}</td>
          <td style="font-weight:600;color:#e2e8f0">{{ t.symbol }}</td>
          <td><span class="badge badge-{{ t.direction|lower }}">{{ t.direction }}</span></td>
          <td>{{ t.entry_date }}</td>
          <td>{{ t.exit_date }}</td>
          <td>${{ '{:,.2f}'.format(t.entry_price) }}</td>
          <td>${{ '{:,.2f}'.format(t.exit_price) }}</td>
          <td>${{ '{:,.0f}'.format(t.size) }}</td>
          <td class="{{ 'green' if t.pnl >= 0 else 'red' }}" style="font-weight:700">
            ${{ '%+,.2f'|format(t.pnl) }}</td>
          <td class="{{ 'green' if t.pnl_pct >= 0 else 'red' }}">{{ '%+.1f'|format(t.pnl_pct) }}%</td>
          <td>{{ '%.2f'|format(t.signal_strength) }}</td>
          <td><span class="badge badge-{{ 'trending' if t.regime=='Trending' else 'mr' if t.regime=='Mean-Rev' else 'neutral' }}">{{ t.regime }}</span></td>
          <td>{{ t.exit_reason }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

</div><!-- container -->

<script>
const equity = {{ equity_json }};
const trades = {{ trades_json }};

// Equity Chart
(function(){
  var labels=equity.map(e=>e.date), data=equity.map(e=>e.capital);
  var ctx=document.getElementById('equityChart').getContext('2d');
  new Chart(ctx,{type:'line',data:{labels:labels,datasets:[
    {label:'Portfolio Value',data:data,borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,0.08)',
     fill:true,tension:0.3,pointRadius:0,borderWidth:2}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b',maxTicksToShow:12},grid:{color:'#1e2744'}},
            y:{ticks:{color:'#64748b',callback:v=>'$'+v.toLocaleString()},grid:{color:'#1e2744'}}}}});
})();

// Drawdown Chart
(function(){
  var labels=equity.map(e=>e.date), caps=equity.map(e=>e.capital);
  var peak=caps[0], dd=caps.map(c=>{peak=Math.max(peak,c);return ((c-peak)/peak)*100;});
  var ctx=document.getElementById('ddChart').getContext('2d');
  new Chart(ctx,{type:'line',data:{labels:labels,datasets:[
    {label:'Drawdown %',data:dd,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.1)',
     fill:true,tension:0.3,pointRadius:0,borderWidth:1.5}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b',maxTicksToShow:8},grid:{color:'#1e2744'}},
            y:{ticks:{color:'#64748b',callback:v=>v.toFixed(0)+'%'},grid:{color:'#1e2744'}}}}});
})();

// P&L Histogram
(function(){
  var pnls=trades.map(t=>t.pnl);
  var mn=Math.min(...pnls),mx=Math.max(...pnls),bins=20;
  var step=(mx-mn)/bins,labels=[],counts=[],colors=[];
  for(var i=0;i<bins;i++){
    var lo=mn+i*step,hi=lo+step;
    labels.push('$'+lo.toFixed(0));
    counts.push(pnls.filter(p=>p>=lo&&p<hi).length);
    colors.push(lo+step/2>=0?'rgba(34,197,94,0.7)':'rgba(239,68,68,0.7)');
  }
  var ctx=document.getElementById('pnlHist').getContext('2d');
  new Chart(ctx,{type:'bar',data:{labels:labels,datasets:[
    {label:'Trades',data:counts,backgroundColor:colors,borderWidth:0}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b',maxRotation:45},grid:{display:false}},
            y:{ticks:{color:'#64748b'},grid:{color:'#1e2744'}}}}});
})();

// Trade filter
function filterTrades(f,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.trade-row').forEach(r=>{
    if(f==='all')r.style.display='';
    else if(f==='win')r.style.display=r.dataset.won==='win'?'':'none';
    else if(f==='loss')r.style.display=r.dataset.won==='loss'?'':'none';
    else if(f==='long')r.style.display=r.dataset.dir==='long'?'':'none';
    else if(f==='short')r.style.display=r.dataset.dir==='short'?'':'none';
  });
}
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(
        DASHBOARD_HTML,
        total_return=_results.get("total_return_pct", 0),
        initial=_results.get("initial_capital", 10000),
        final=_results.get("final_capital", 0),
        sharpe=_results.get("sharpe_ratio", 0),
        max_dd=_results.get("max_drawdown_pct", 0),
        win_rate=_results.get("win_rate", 0),
        n_wins=_results.get("n_wins", 0),
        n_losses=_results.get("n_losses", 0),
        pf=_results.get("profit_factor", 0),
        n_trades=_results.get("n_trades", 0),
        n_folds=_results.get("n_folds", 0),
        avg_win=_results.get("avg_win", 0),
        avg_loss=abs(_results.get("avg_loss", 0)),
        trades=_trades,
        equity_json=json.dumps(_equity),
        trades_json=json.dumps(_trades),
        folds=_folds,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Dashboard ready at http://localhost:8050")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
