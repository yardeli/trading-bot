"""
Backtest Metrics — Comprehensive performance analysis.
"""
import numpy as np
import pandas as pd


def compute_metrics(equity_curve: list[dict], trades: list) -> dict:
    """Compute comprehensive trading metrics."""
    if not trades:
        return {}

    eq = pd.DataFrame(equity_curve)
    daily_returns = eq["capital"].pct_change().dropna()

    # Basic stats
    n_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    # Return metrics
    total_pnl = sum(t.pnl for t in trades)
    total_staked = sum(t.size for t in trades)

    # Risk metrics
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)
              if daily_returns.std() > 0 else 0)

    sortino_denom = daily_returns[daily_returns < 0].std()
    sortino = (daily_returns.mean() / sortino_denom * np.sqrt(365)
               if sortino_denom > 0 else 0)

    # Drawdown
    peak = eq["capital"].expanding().max()
    drawdowns = (eq["capital"] - peak) / peak
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

    # Calmar ratio
    calmar = ((eq["capital"].iloc[-1] / eq["capital"].iloc[0] - 1) / max_dd
              if max_dd > 0 else 0)

    # Profit factor
    gross_wins = sum(t.pnl for t in wins) if wins else 0
    gross_losses = abs(sum(t.pnl for t in losses)) if losses else 0.01

    # Win/loss streaks
    results = [1 if t.pnl > 0 else 0 for t in trades]
    max_win_streak = _max_streak(results, 1)
    max_loss_streak = _max_streak(results, 0)

    # Average holding period
    durations = []
    for t in trades:
        if t.entry_date and t.exit_date:
            dur = (t.exit_date - t.entry_date).days
            durations.append(dur)

    # Trade analysis by regime
    regime_stats = {}
    for regime_val in [-1, 0, 1]:
        regime_trades = [t for t in trades if t.regime == regime_val]
        if regime_trades:
            r_wins = [t for t in regime_trades if t.pnl > 0]
            regime_stats[regime_val] = {
                "n_trades": len(regime_trades),
                "win_rate": round(len(r_wins) / len(regime_trades) * 100, 1),
                "total_pnl": round(sum(t.pnl for t in regime_trades), 2),
            }

    return {
        "n_trades": n_trades,
        "win_rate": round(len(wins) / max(n_trades, 1) * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / max(total_staked, 1) * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "profit_factor": round(gross_wins / max(gross_losses, 0.01), 3),
        "avg_win": round(sum(t.pnl for t in wins) / max(len(wins), 1), 2),
        "avg_loss": round(sum(t.pnl for t in losses) / max(len(losses), 1), 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_holding_days": round(np.mean(durations), 1) if durations else 0,
        "regime_stats": regime_stats,
    }


def _max_streak(results: list[int], target: int) -> int:
    max_s = 0
    current = 0
    for r in results:
        if r == target:
            current += 1
            max_s = max(max_s, current)
        else:
            current = 0
    return max_s


def print_results(results: dict):
    """Pretty-print backtest results."""
    print("\n" + "=" * 60)
    print("  AUTO-TRADER BACKTEST RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    print(f"  Initial Capital:  ${results.get('initial_capital', 0):,.0f}")
    print(f"  Final Capital:    ${results.get('final_capital', 0):,.0f}")
    print(f"  Total Return:     {results.get('total_return_pct', 0):+.1f}%")
    print(f"  ROI:              {results.get('roi_pct', 0):+.1f}%")
    print()
    print(f"  Trades:           {results.get('n_trades', 0)}")
    print(f"  Win Rate:         {results.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:    {results.get('profit_factor', 0):.2f}")
    print(f"  Avg Win:          ${results.get('avg_win', 0):,.2f}")
    print(f"  Avg Loss:         ${results.get('avg_loss', 0):,.2f}")
    print()
    print(f"  Sharpe Ratio:     {results.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:     {results.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Folds:            {results.get('n_folds', 0)}")
    print("=" * 60)
