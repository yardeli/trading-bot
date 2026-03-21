"""
Rich Terminal Dashboard for the Quantitative Trading Engine.

Renders a live-updating terminal UI showing:
    - Equity curve (ASCII sparkline)
    - Performance metrics (Sharpe, return, drawdown, etc.)
    - Current portfolio positions & weights
    - Risk metrics (VaR, exposure, vol)
    - Recent trade activity log
    - Alpha model weights from ensemble
    - Progress bar during backtest simulation
"""
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


# ── ASCII Sparkline ──────────────────────────────────────────────
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values or len(values) < 2:
        return ""
    # Downsample to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi != lo else 1.0
    chars = []
    for v in sampled:
        idx = int((v - lo) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def colored_sparkline(values: list[float], width: int = 60) -> Text:
    """Sparkline with green for gains, red for losses vs start."""
    if not values or len(values) < 2:
        return Text("")
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi != lo else 1.0
    start = sampled[0]
    text = Text()
    for v in sampled:
        idx = int((v - lo) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        color = "green" if v >= start else "red"
        text.append(SPARK_CHARS[idx], style=color)
    return text


def drawdown_sparkline(values: list[float], width: int = 60) -> Text:
    """Sparkline for drawdown (always red, deeper = darker)."""
    if not values or len(values) < 2:
        return Text("")
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi != lo else 1.0
    text = Text()
    # Invert: drawdown is negative, show magnitude
    for v in sampled:
        mag = abs(v)
        idx = int(mag / (abs(lo) if lo != 0 else 1.0) * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        text.append(SPARK_CHARS[idx], style="red")
    return text


# ── Dashboard ────────────────────────────────────────────────────


class TerminalDashboard:
    """
    Live terminal dashboard for backtest progress and results.

    Usage:
        dashboard = TerminalDashboard()
        dashboard.on_backtest_start(total_days, config_info)
        for each day:
            dashboard.on_day_update(date, equity, ...)
        dashboard.on_backtest_complete(result)
        dashboard.show_final_report(result)
    """

    def __init__(self):
        # Reconfigure stdout to UTF-8 on Windows to support Unicode sparkline characters
        if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        self.console = Console()
        self.live: Optional[Live] = None
        self.progress: Optional[Progress] = None
        self.task_id = None

        # State
        self.equity_history: list[float] = []
        self.date_history: list[str] = []
        self.return_history: list[float] = []
        self.drawdown_history: list[float] = []
        self.peak_equity = 0.0

        self.current_weights: dict[str, float] = {}
        self.risk_metrics: dict = {}
        self.model_weights: dict[str, float] = {}
        self.recent_trades: list[dict] = []
        self.config_info: dict = {}
        self.initial_capital = 1_000_000

        # Running performance
        self.total_return = 0.0
        self.ann_return = 0.0
        self.ann_vol = 0.0
        self.sharpe = 0.0
        self.max_dd = 0.0
        self.win_rate = 0.0
        self.current_dd = 0.0

    # ── Lifecycle ────────────────────────────────────────────────

    def on_backtest_start(self, total_days: int, config_info: dict, initial_capital: float):
        """Called when the backtest begins."""
        self.config_info = config_info
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.task_id = self.progress.add_task("Backtesting...", total=total_days)

        self.live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self.live.start()

    def on_day_update(
        self,
        date,
        equity: float,
        weights: Optional[dict] = None,
        risk_metrics: Optional[dict] = None,
        model_weights: Optional[dict] = None,
        trade_info: Optional[dict] = None,
    ):
        """Called each simulated day."""
        self.equity_history.append(equity)
        date_str = str(date)[:10] if hasattr(date, 'strftime') else str(date)[:10]
        self.date_history.append(date_str)

        # Track returns
        if len(self.equity_history) >= 2:
            prev = self.equity_history[-2]
            ret = (equity - prev) / prev if prev != 0 else 0.0
            self.return_history.append(ret)
        else:
            self.return_history.append(0.0)

        # Track drawdown
        self.peak_equity = max(self.peak_equity, equity)
        dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.drawdown_history.append(dd)
        self.current_dd = dd

        # Update running metrics
        self._update_running_metrics(equity)

        if weights is not None:
            self.current_weights = weights
        if risk_metrics is not None:
            self.risk_metrics = risk_metrics
        if model_weights is not None:
            self.model_weights = model_weights
        if trade_info is not None:
            self.recent_trades.append(trade_info)
            if len(self.recent_trades) > 8:
                self.recent_trades = self.recent_trades[-8:]

        # Advance progress
        if self.progress and self.task_id is not None:
            self.progress.advance(self.task_id)

        # Update display
        if self.live:
            self.live.update(self._build_layout())

    def on_backtest_complete(self):
        """Called when the backtest finishes."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description="[bold green]Complete!")
        if self.live:
            self.live.update(self._build_layout())
            time.sleep(0.5)
            self.live.stop()

    def show_final_report(self, result):
        """Display the final results dashboard (static)."""
        self.console.print()
        self.console.print(self._build_final_report(result))
        self.console.print()

    # ── Running metrics ──────────────────────────────────────────

    def _update_running_metrics(self, equity: float):
        self.total_return = (equity / self.initial_capital) - 1
        n_days = len(self.equity_history)
        n_years = n_days / 252

        if n_years > 0.01:
            self.ann_return = (1 + self.total_return) ** (1 / n_years) - 1

        if len(self.return_history) > 20:
            rets = np.array(self.return_history[-252:])
            self.ann_vol = float(np.std(rets) * np.sqrt(252))
            if self.ann_vol > 0:
                self.sharpe = self.ann_return / self.ann_vol

            wins = np.sum(np.array(rets) > 0)
            self.win_rate = wins / len(rets)

        self.max_dd = min(self.drawdown_history) if self.drawdown_history else 0.0

    # ── Layout builders ──────────────────────────────────────────

    def _build_layout(self) -> Panel:
        """Build the full dashboard layout."""
        sections = []

        # Header
        sections.append(self._build_header())

        # Progress bar
        if self.progress:
            sections.append(self.progress)

        # Equity sparkline
        if len(self.equity_history) > 2:
            sections.append(self._build_equity_panel())

        # Main metrics + positions side by side
        cols = []
        cols.append(self._build_metrics_panel())
        if self.current_weights:
            cols.append(self._build_positions_panel())
        if cols:
            sections.append(Columns(cols, expand=True, equal=True))

        # Risk + Models + Trades
        bottom_cols = []
        if self.risk_metrics:
            bottom_cols.append(self._build_risk_panel())
        if self.model_weights:
            bottom_cols.append(self._build_model_weights_panel())
        if self.recent_trades:
            bottom_cols.append(self._build_trades_panel())
        if bottom_cols:
            sections.append(Columns(bottom_cols, expand=True, equal=True))

        return Panel(
            Group(*sections),
            title="[bold white on blue]  QUANTITATIVE TRADING ENGINE  [/]",
            border_style="blue",
            padding=(1, 2),
        )

    def _build_header(self) -> Text:
        """Config summary header."""
        c = self.config_info
        text = Text()
        text.append("  Universe: ", style="dim")
        text.append(f"{c.get('n_assets', '?')} assets", style="bold")
        text.append("  |  History: ", style="dim")
        text.append(f"{c.get('years', '?')}y", style="bold")
        text.append("  |  Method: ", style="dim")
        text.append(f"{c.get('method', '?')}", style="bold cyan")
        text.append("  |  Ensemble: ", style="dim")
        text.append(f"{c.get('ensemble', '?')}", style="bold cyan")
        text.append("  |  Vol Target: ", style="dim")
        text.append(f"{c.get('vol_target', 0):.0%}", style="bold yellow")
        text.append("\n")
        return text

    def _build_equity_panel(self) -> Panel:
        """Equity curve sparkline + drawdown."""
        eq = self.equity_history
        current = eq[-1] if eq else self.initial_capital
        pnl = current - self.initial_capital
        pnl_pct = self.total_return

        title_parts = Text()
        title_parts.append(f"  Equity: ${current:,.0f}", style="bold")
        title_parts.append(f"  (", style="dim")
        color = "green" if pnl >= 0 else "red"
        sign = "+" if pnl >= 0 else ""
        title_parts.append(f"{sign}${pnl:,.0f}", style=f"bold {color}")
        title_parts.append(f" / {sign}{pnl_pct:.2%}", style=color)
        title_parts.append(f")", style="dim")

        content = Text()
        content.append("  Equity  ")
        content.append_text(colored_sparkline(eq, width=70))
        content.append(f"\n  Drawdown")
        content.append(" ")
        content.append_text(drawdown_sparkline(self.drawdown_history, width=70))

        # Date range
        if self.date_history:
            content.append(f"\n  {self.date_history[0]}", style="dim")
            content.append(" " * max(0, 58 - len(self.date_history[0]) - len(self.date_history[-1])))
            content.append(f"{self.date_history[-1]}", style="dim")

        return Panel(content, title=title_parts, border_style="green" if pnl >= 0 else "red")

    def _build_metrics_panel(self) -> Panel:
        """Performance metrics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", justify="right", width=12)

        def _color_val(val, fmt=".2%", good_positive=True):
            s = f"{val:{fmt}}"
            if good_positive:
                return f"[green]{s}[/]" if val > 0 else f"[red]{s}[/]"
            else:
                return f"[red]{s}[/]" if val < 0 else f"[green]{s}[/]"

        table.add_row("Total Return", _color_val(self.total_return))
        table.add_row("Ann. Return", _color_val(self.ann_return))
        table.add_row("Ann. Volatility", f"{self.ann_vol:.2%}")
        table.add_row("Sharpe Ratio", _color_val(self.sharpe, ".2f"))
        table.add_row("Max Drawdown", _color_val(self.max_dd, ".2%", good_positive=False))
        table.add_row("Current DD", _color_val(self.current_dd, ".2%", good_positive=False))
        table.add_row("Win Rate", f"{self.win_rate:.1%}")
        table.add_row("Trading Days", f"{len(self.equity_history)}")

        return Panel(table, title="[bold]Performance", border_style="cyan")

    def _build_positions_panel(self) -> Panel:
        """Current portfolio positions."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Asset", style="bold", width=8)
        table.add_column("Weight", justify="right", width=8)
        table.add_column("Bar", width=14)

        # Sort by absolute weight, show top positions
        sorted_w = sorted(self.current_weights.items(), key=lambda x: -abs(x[1]))
        for asset, weight in sorted_w[:12]:
            if abs(weight) < 0.001:
                continue
            color = "green" if weight > 0 else "red"
            bar_len = int(abs(weight) * 40)
            bar = ("+" if weight > 0 else "-") * min(bar_len, 14)
            table.add_row(
                asset,
                f"[{color}]{weight:+.1%}[/]",
                f"[{color}]{bar}[/]",
            )

        n_active = sum(1 for w in self.current_weights.values() if abs(w) > 0.001)
        gross = sum(abs(w) for w in self.current_weights.values())
        net = sum(self.current_weights.values())

        footer = Text()
        footer.append(f"\n  {n_active} positions", style="dim")
        footer.append(f"  G:{gross:.2f}  N:{net:+.2f}", style="dim")

        return Panel(Group(table, footer), title="[bold]Positions", border_style="yellow")

    def _build_risk_panel(self) -> Panel:
        """Risk metrics display."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim", width=16)
        table.add_column("Value", justify="right", width=10)

        rm = self.risk_metrics
        if "var_1d" in rm:
            table.add_row("VaR (1d, 99%)", f"{rm['var_1d']:.3%}")
        if "expected_shortfall" in rm:
            table.add_row("Exp. Shortfall", f"{rm['expected_shortfall']:.3%}")
        if "realized_vol" in rm:
            table.add_row("Realized Vol", f"{rm['realized_vol']:.1%}")
        if "vol_scale" in rm:
            table.add_row("Vol Scale", f"{rm['vol_scale']:.2f}x")
        if "gross_exposure" in rm:
            table.add_row("Gross Exp.", f"{rm['gross_exposure']:.2f}")
        if "net_exposure" in rm:
            table.add_row("Net Exp.", f"{rm['net_exposure']:+.2f}")
        if "herfindahl" in rm:
            table.add_row("HHI (conc.)", f"{rm['herfindahl']:.3f}")

        return Panel(table, title="[bold]Risk", border_style="red")

    def _build_model_weights_panel(self) -> Panel:
        """Alpha model ensemble weights."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Model", style="bold", width=20)
        table.add_column("Weight", justify="right", width=8)
        table.add_column("Bar", width=12)

        sorted_mw = sorted(self.model_weights.items(), key=lambda x: -(x[1] if x[1] == x[1] else 0))
        for name, weight in sorted_mw:
            if weight != weight:  # NaN check
                weight = 0.0
            bar_len = int(weight * 30)
            bar = "#" * min(bar_len, 12)
            table.add_row(name, f"{weight:.1%}", f"[cyan]{bar}[/]")

        return Panel(table, title="[bold]Model Weights", border_style="magenta")

    def _build_trades_panel(self) -> Panel:
        """Recent trade activity."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Date", style="dim", width=10)
        table.add_column("#", justify="right", width=3)
        table.add_column("Turnover", justify="right", width=8)
        table.add_column("Cost", justify="right", width=10)

        for trade in self.recent_trades[-6:]:
            table.add_row(
                str(trade.get("date", ""))[:10],
                str(trade.get("n_trades", 0)),
                f"{trade.get('turnover', 0):.2%}",
                f"${trade.get('total_cost', 0):,.0f}",
            )

        return Panel(table, title="[bold]Recent Trades", border_style="white")

    # ── Final Report ─────────────────────────────────────────────

    def _build_final_report(self, result) -> Panel:
        """Static final report after backtest completes."""
        m = result.performance_metrics
        ex = result.execution_summary
        if not m:
            return Panel("No results to display.", title="Results")

        # Top section: big numbers
        header = Text()
        header.append("\n")
        header.append("  FINAL EQUITY   ", style="dim")
        header.append(f"${m['final_equity']:,.2f}\n", style="bold white")
        header.append("  TOTAL RETURN   ", style="dim")
        color = "green" if m['total_return'] >= 0 else "red"
        header.append(f"{m['total_return']:+.2%}\n", style=f"bold {color}")
        header.append("  SHARPE RATIO   ", style="dim")
        color = "green" if m['sharpe_ratio'] > 0 else "red"
        header.append(f"{m['sharpe_ratio']:.2f}\n\n", style=f"bold {color}")

        # Equity sparkline
        eq_vals = result.equity_curve.values.tolist()
        spark_text = Text("  ")
        spark_text.append_text(colored_sparkline(eq_vals, width=80))
        spark_text.append("\n")

        # Performance table
        perf_table = Table(title="Performance Metrics", show_header=True, border_style="cyan")
        perf_table.add_column("Metric", style="bold", width=22)
        perf_table.add_column("Value", justify="right", width=14)

        def _fmt(val, fmt=".2%"):
            return f"{val:{fmt}}"

        perf_table.add_row("Annualized Return", _fmt(m["annualized_return"]))
        perf_table.add_row("Annualized Volatility", _fmt(m["annualized_volatility"]))
        perf_table.add_row("Sharpe Ratio", _fmt(m["sharpe_ratio"], ".2f"))
        perf_table.add_row("Sortino Ratio", _fmt(m["sortino_ratio"], ".2f"))
        perf_table.add_row("Max Drawdown", _fmt(m["max_drawdown"]))
        perf_table.add_row("Calmar Ratio", _fmt(m["calmar_ratio"], ".2f"))
        perf_table.add_row("Win Rate", _fmt(m["win_rate"]))
        perf_table.add_row("Profit Factor", _fmt(m["profit_factor"], ".2f"))
        perf_table.add_row("Skewness", _fmt(m["skewness"], ".3f"))
        perf_table.add_row("Kurtosis", _fmt(m["kurtosis"], ".3f"))
        perf_table.add_row("Trading Days", str(m["n_trading_days"]))

        # Execution table
        exec_table = Table(title="Execution Summary", show_header=True, border_style="yellow")
        exec_table.add_column("Metric", style="bold", width=22)
        exec_table.add_column("Value", justify="right", width=14)

        exec_table.add_row("Total Costs", f"${ex['total_costs']:,.2f}")
        exec_table.add_row("Total Turnover", f"{ex['total_turnover']:.2f}x")
        exec_table.add_row("Rebalances", str(ex["n_rebalances"]))
        exec_table.add_row("Avg Cost/Rebal", f"${ex['avg_cost_per_rebalance']:,.2f}")
        exec_table.add_row("Avg Turnover/Rebal", f"{ex['avg_turnover']:.2%}")

        tables = Columns([perf_table, exec_table], expand=True, equal=True)

        return Panel(
            Group(header, spark_text, tables),
            title="[bold white on blue]  BACKTEST RESULTS  [/]",
            border_style="blue",
            padding=(1, 2),
        )
