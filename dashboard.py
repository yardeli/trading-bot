"""
Auto-Trader Dashboard — Local web UI for backtest visualization.
Run: python dashboard.py
Then open http://localhost:8050
"""
import os
import sys
import json
from datetime import datetime

from flask import Flask, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)


def run_backtest_with_details():
    """Run a fresh backtest and capture detailed trade/equity data."""
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

    trades = []
    for t in bt.trades:
        trades.append({
            "symbol": t.symbol,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_date": str(t.entry_date)[:10] if t.entry_date else "",
            "exit_date": str(t.exit_date)[:10] if t.exit_date else "",
            "entry_price": float(round(t.entry_price, 2)) if t.entry_price else 0,
            "exit_price": float(round(t.exit_price, 2)) if t.exit_price else 0,
            "size": float(round(t.size, 2)),
            "pnl": float(round(t.pnl, 2)),
            "pnl_pct": float(round(t.pnl_pct * 100, 2)),
            "signal_strength": float(round(t.signal_strength, 3)),
            "regime": {1: "Trending", -1: "Mean-Rev", 0: "Neutral"}.get(t.regime, "?"),
            "exit_reason": t.exit_reason,
            "won": bool(t.pnl > 0),
        })

    equity = [{"date": str(e["date"])[:10], "capital": float(round(e["capital"], 2)),
               "fold": int(e["fold"]), "positions": int(e["open_positions"])}
              for e in bt.equity_curve]

    # Sanitize fold results
    clean_folds = []
    for f in bt.fold_results:
        clean_folds.append({k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in f.items()})

    # Sanitize results
    clean_results = {}
    for k, v in results.items():
        if k == "fold_results":
            clean_results[k] = clean_folds
        elif isinstance(v, (int, float)):
            clean_results[k] = float(v)
        else:
            clean_results[k] = v

    return clean_results, trades, equity, clean_folds


# Cache results at startup
print("=" * 50)
print("  Auto-Trader Dashboard")
print("  Running backtest, please wait...")
print("=" * 50)
_results, _trades, _equity, _folds = run_backtest_with_details()


def _build_html():
    """Build dashboard HTML with all data embedded as JSON."""
    data_json = json.dumps({
        "results": _results,
        "trades": _trades,
        "equity": _equity,
        "folds": _folds,
    })
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auto-Trader Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0a0e1a;color:#c8d0e0}
.header{background:linear-gradient(135deg,#0f1629,#1a1f3a);padding:20px 30px;
  border-bottom:1px solid #1e2744;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:1.5rem;color:#e2e8f0}
.header h1 span{color:#22d3ee}
.header .ts{color:#64748b;font-size:.8rem}
.container{max-width:1400px;margin:0 auto;padding:20px}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:22px}
.sc{background:#111827;border:1px solid #1e2744;border-radius:10px;padding:14px;text-align:center}
.sc:hover{border-color:#334155}
.sl{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.sv{font-size:1.5rem;font-weight:700}
.ss{font-size:.7rem;color:#64748b;margin-top:3px}
.g{color:#22c55e}.r{color:#ef4444}.b{color:#3b82f6}.cy{color:#22d3ee}.or{color:#f97316}
.cc{background:#111827;border:1px solid #1e2744;border-radius:10px;padding:20px;margin-bottom:22px}
.cc h2{font-size:1rem;color:#94a3b8;margin-bottom:12px}
.cw{height:340px;position:relative}
.sec{background:#111827;border:1px solid #1e2744;border-radius:10px;padding:20px;margin-bottom:22px}
.sec h2{font-size:1rem;color:#94a3b8;margin-bottom:12px}
.tabs{display:flex;gap:8px;margin-bottom:14px}
.tab{padding:5px 14px;border-radius:6px;cursor:pointer;font-size:.78rem;
  background:#1e293b;color:#94a3b8;border:1px solid #334155}
.tab.active{background:#22d3ee;color:#0a0e1a;border-color:#22d3ee;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.76rem}
th{background:#1e293b;color:#94a3b8;padding:7px 8px;text-align:left;
  font-weight:600;text-transform:uppercase;letter-spacing:.5px;font-size:.66rem;position:sticky;top:0}
td{padding:6px 8px;border-bottom:1px solid #1e2744}
tr:hover{background:#1a2235}
.badge{padding:2px 7px;border-radius:4px;font-size:.66rem;font-weight:600}
.badge-long{background:#064e3b;color:#34d399}
.badge-short{background:#7f1d1d;color:#f87171}
.badge-trending{background:#1e3a5f;color:#60a5fa}
.badge-mr{background:#3b1f5e;color:#c084fc}
.badge-neutral{background:#374151;color:#9ca3af}
.stbl{max-height:500px;overflow-y:auto}
.tcol{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.tcol{grid-template-columns:1fr}}
.fc{background:#0f1629;border:1px solid #1e2744;border-radius:8px;padding:10px 14px;
  display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;font-size:.78rem}
.fc .fi{font-weight:700;color:#22d3ee;width:55px}
.fc .fs{display:flex;gap:14px}
</style>
</head>
<body>
<div class="header">
  <h1><span>&#9883;</span> Auto-Trader <span>Dashboard</span></h1>
  <div class="ts">Backtest completed __TIMESTAMP__</div>
</div>
<div class="container">
<div class="sg" id="stats-grid"></div>
<div class="cc"><h2>Equity Curve &mdash; Walk-Forward Backtest</h2><div class="cw"><canvas id="eqChart"></canvas></div></div>
<div class="tcol">
  <div class="cc"><h2>Drawdown</h2><div class="cw"><canvas id="ddChart"></canvas></div></div>
  <div class="cc"><h2>Trade P&amp;L Distribution</h2><div class="cw"><canvas id="pnlHist"></canvas></div></div>
</div>
<div class="sec"><h2>Walk-Forward Fold Results</h2><div id="folds-container"></div></div>
<div class="sec">
  <h2 id="trade-title">Trade Log</h2>
  <div class="tabs">
    <div class="tab active" onclick="filterT('all',this)">All</div>
    <div class="tab" onclick="filterT('win',this)">Winners</div>
    <div class="tab" onclick="filterT('loss',this)">Losers</div>
    <div class="tab" onclick="filterT('long',this)">Longs</div>
    <div class="tab" onclick="filterT('short',this)">Shorts</div>
  </div>
  <div class="stbl"><table><thead><tr>
    <th>#</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
    <th>Entry $</th><th>Exit $</th><th>Size</th><th>P&amp;L</th><th>P&amp;L%</th>
    <th>Signal</th><th>Regime</th><th>Exit Reason</th>
  </tr></thead><tbody id="tbody"></tbody></table></div>
</div>
</div>
<script>
var D=__DATA_JSON__;
var R=D.results,T=D.trades,E=D.equity,F=D.folds;
function $(s){return document.getElementById(s)}
function fmt(n){return n.toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:0})}
function fmt2(n){return n.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}

// Stats
var stats=[
  ['Total Return',(R.total_return_pct>=0?'+':'')+R.total_return_pct.toFixed(1)+'%',
   R.total_return_pct>=0?'g':'r','$'+fmt(R.initial_capital)+' → $'+fmt(R.final_capital)],
  ['Sharpe Ratio',R.sharpe_ratio.toFixed(3),R.sharpe_ratio>=1?'g':R.sharpe_ratio>=0.5?'or':'r','Risk-adjusted'],
  ['Max Drawdown',R.max_drawdown_pct.toFixed(1)+'%','r','Peak to trough'],
  ['Win Rate',R.win_rate.toFixed(1)+'%',R.win_rate>=50?'g':'or',R.n_wins+'W / '+R.n_losses+'L'],
  ['Profit Factor',R.profit_factor.toFixed(2),R.profit_factor>=1.2?'g':'or','Gross win/loss'],
  ['Total Trades',R.n_trades,'cy',R.n_folds+' folds'],
  ['Avg Win','$'+fmt(R.avg_win),'g','Per winning trade'],
  ['Avg Loss','$'+fmt(Math.abs(R.avg_loss)),'r','Per losing trade'],
];
var sh='';stats.forEach(function(s){
  sh+='<div class="sc"><div class="sl">'+s[0]+'</div><div class="sv '+s[2]+'">'+s[1]+'</div><div class="ss">'+s[3]+'</div></div>';
});$('stats-grid').innerHTML=sh;

// Equity chart
(function(){
  var ctx=$('eqChart').getContext('2d');
  new Chart(ctx,{type:'line',data:{labels:E.map(e=>e.date),datasets:[
    {label:'Portfolio',data:E.map(e=>e.capital),borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,0.08)',
     fill:true,tension:.3,pointRadius:0,borderWidth:2}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b',maxTicksToShow:12},grid:{color:'#1e2744'}},
            y:{ticks:{color:'#64748b',callback:v=>'$'+v.toLocaleString()},grid:{color:'#1e2744'}}}}});
})();

// Drawdown
(function(){
  var caps=E.map(e=>e.capital),peak=caps[0];
  var dd=caps.map(c=>{peak=Math.max(peak,c);return ((c-peak)/peak)*100});
  var ctx=$('ddChart').getContext('2d');
  new Chart(ctx,{type:'line',data:{labels:E.map(e=>e.date),datasets:[
    {label:'DD%',data:dd,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.1)',
     fill:true,tension:.3,pointRadius:0,borderWidth:1.5}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b',maxTicksToShow:8},grid:{color:'#1e2744'}},
            y:{ticks:{color:'#64748b',callback:v=>v.toFixed(0)+'%'},grid:{color:'#1e2744'}}}}});
})();

// P&L Histogram
(function(){
  var pnls=T.map(t=>t.pnl);if(!pnls.length)return;
  var mn=Math.min(...pnls),mx=Math.max(...pnls),bins=20,step=(mx-mn)/bins;
  var labels=[],counts=[],colors=[];
  for(var i=0;i<bins;i++){
    var lo=mn+i*step;labels.push('$'+lo.toFixed(0));
    counts.push(pnls.filter(p=>p>=lo&&p<lo+step).length);
    colors.push(lo+step/2>=0?'rgba(34,197,94,0.7)':'rgba(239,68,68,0.7)');
  }
  var ctx=$('pnlHist').getContext('2d');
  new Chart(ctx,{type:'bar',data:{labels:labels,datasets:[
    {label:'Trades',data:counts,backgroundColor:colors,borderWidth:0}
  ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b',maxRotation:45},grid:{display:false}},
            y:{ticks:{color:'#64748b'},grid:{color:'#1e2744'}}}}});
})();

// Folds
var fh='';F.forEach(function(f){
  var rc=f.roi_pct>=0?'g':'r',wc=f.win_rate>=50?'g':'or';
  fh+='<div class="fc"><div class="fi">Fold '+f.fold+'</div><div class="fs">'
    +'<span>'+f.n_trades+' trades</span>'
    +'<span class="'+wc+'">'+f.win_rate.toFixed(0)+'% win</span>'
    +'<span class="'+rc+'">'+(f.roi_pct>=0?'+':'')+f.roi_pct.toFixed(1)+'% ROI</span>'
    +'<span>$'+fmt(f.starting_capital)+' → $'+fmt(f.ending_capital)+'</span>'
    +'</div></div>';
});$('folds-container').innerHTML=fh;

// Trades table
$('trade-title').textContent='Trade Log ('+T.length+' trades)';
var tb='';T.forEach(function(t,i){
  var dc=t.direction==='LONG'?'long':'short';
  var pc=t.pnl>=0?'g':'r';
  var rc=t.regime==='Trending'?'trending':t.regime==='Mean-Rev'?'mr':'neutral';
  tb+='<tr class="trow" data-won="'+(t.won?'win':'loss')+'" data-dir="'+dc+'">'
    +'<td>'+(i+1)+'</td>'
    +'<td style="font-weight:600;color:#e2e8f0">'+t.symbol+'</td>'
    +'<td><span class="badge badge-'+dc+'">'+t.direction+'</span></td>'
    +'<td>'+t.entry_date+'</td><td>'+t.exit_date+'</td>'
    +'<td>$'+fmt2(t.entry_price)+'</td><td>$'+fmt2(t.exit_price)+'</td>'
    +'<td>$'+fmt(t.size)+'</td>'
    +'<td class="'+pc+'" style="font-weight:700">$'+(t.pnl>=0?'+':'')+fmt2(t.pnl)+'</td>'
    +'<td class="'+pc+'">'+(t.pnl_pct>=0?'+':'')+t.pnl_pct.toFixed(1)+'%</td>'
    +'<td>'+t.signal_strength.toFixed(2)+'</td>'
    +'<td><span class="badge badge-'+rc+'">'+t.regime+'</span></td>'
    +'<td>'+t.exit_reason+'</td></tr>';
});$('tbody').innerHTML=tb;

function filterT(f,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.trow').forEach(r=>{
    if(f==='all')r.style.display='';
    else if(f==='win')r.style.display=r.dataset.won==='win'?'':'none';
    else if(f==='loss')r.style.display=r.dataset.won==='loss'?'':'none';
    else if(f==='long')r.style.display=r.dataset.dir==='long'?'':'none';
    else if(f==='short')r.style.display=r.dataset.dir==='short'?'':'none';
  });
}
</script>
</body>
</html>""".replace("__DATA_JSON__", data_json).replace("__TIMESTAMP__", timestamp)


@app.route("/")
def dashboard():
    html = _build_html()
    return Response(html, content_type="text/html")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Dashboard ready at http://localhost:8050")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
