"""
Walk-Forward Backtester with CPCV support.
Simulates realistic trading with regime-adaptive signals and Kelly sizing.
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WALK_FORWARD_TRAIN_DAYS, WALK_FORWARD_TEST_DAYS, WALK_FORWARD_STEP_DAYS,
    CPCV_N_SPLITS, CPCV_N_TEST_SPLITS, PURGE_DAYS, MIN_TRADES_PER_FOLD,
    INITIAL_CAPITAL, MAX_POSITION_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TRANSACTION_COST_PCT, MAX_DRAWDOWN_HALT, TRAILING_STOP_PCT,
    MAX_PORTFOLIO_EXPOSURE,
)
from models.ensemble import EnsembleModel
from strategy.regime import RegimeDetector
from strategy.signals import combined_signal
from strategy.kelly import kelly_fraction, dynamic_kelly


class Trade:
    __slots__ = ["symbol", "entry_date", "entry_price", "size", "direction",
                 "exit_date", "exit_price", "pnl", "pnl_pct", "signal_strength",
                 "regime", "exit_reason"]

    def __init__(self, symbol, entry_date, entry_price, size, direction,
                 signal_strength=0, regime=0):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.size = size
        self.direction = direction
        self.signal_strength = signal_strength
        self.regime = regime
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.exit_reason = ""


class WalkForwardBacktester:
    """Walk-forward backtester with regime-adaptive signals."""

    def __init__(self):
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self.fold_results: list[dict] = []

    def run(
        self,
        dataset: pd.DataFrame,
        features_dict: dict[str, pd.DataFrame],
        data_dict: dict[str, pd.DataFrame],
    ) -> dict:
        """Run walk-forward backtest."""
        dataset = dataset.sort_values("date").reset_index(drop=True)
        dates = sorted(dataset["date"].unique())

        if len(dates) < WALK_FORWARD_TRAIN_DAYS + WALK_FORWARD_TEST_DAYS:
            raise ValueError(f"Not enough data: {len(dates)} days, need "
                           f"{WALK_FORWARD_TRAIN_DAYS + WALK_FORWARD_TEST_DAYS}")

        capital = INITIAL_CAPITAL
        peak_capital = capital
        fold_idx = 0
        start = 0

        while start + WALK_FORWARD_TRAIN_DAYS + WALK_FORWARD_TEST_DAYS <= len(dates):
            train_end = start + WALK_FORWARD_TRAIN_DAYS
            test_end = min(train_end + WALK_FORWARD_TEST_DAYS, len(dates))

            train_dates = dates[start:train_end]
            test_dates = dates[train_end:test_end]

            train_mask = dataset["date"].isin(train_dates)
            test_mask = dataset["date"].isin(test_dates)

            train_df = dataset[train_mask]
            test_df = dataset[test_mask]

            if len(train_df) < 100 or len(test_df) < 10:
                start += WALK_FORWARD_STEP_DAYS
                continue

            # Train ensemble model
            model = EnsembleModel()
            try:
                model.fit(train_df, "target_up")
            except Exception as e:
                print(f"  [WARN] Fold {fold_idx} train failed: {e}")
                start += WALK_FORWARD_STEP_DAYS
                continue

            # Predict on test set
            probs = model.predict_proba(test_df)
            test_df = test_df.copy()
            test_df["pred_prob"] = probs

            # Simulate trading on test period
            fold_result = self._simulate_fold(
                test_df, features_dict, data_dict, capital, peak_capital, fold_idx,
            )

            capital = fold_result["ending_capital"]
            peak_capital = max(peak_capital, capital)
            self.fold_results.append(fold_result)

            print(f"  Fold {fold_idx}: {fold_result['n_trades']} trades, "
                  f"ROI={fold_result['roi_pct']:.1f}%, "
                  f"Win={fold_result['win_rate']:.1f}%, "
                  f"Capital=${capital:.0f}")

            fold_idx += 1
            start += WALK_FORWARD_STEP_DAYS

            # Check max drawdown halt
            dd = (peak_capital - capital) / peak_capital
            if dd >= MAX_DRAWDOWN_HALT:
                print(f"  [HALT] Max drawdown {dd:.1%} reached")
                break

        return self._compile_results(capital)

    def _simulate_fold(
        self,
        test_df: pd.DataFrame,
        features_dict: dict[str, pd.DataFrame],
        data_dict: dict[str, pd.DataFrame],
        capital: float,
        peak_capital: float,
        fold_idx: int,
    ) -> dict:
        """Simulate trading on a test fold."""
        regime_detector = RegimeDetector()
        fold_trades = []
        open_positions: dict[str, Trade] = {}
        fold_start_capital = capital

        # Group test data by date
        for date in sorted(test_df["date"].unique()):
            day_data = test_df[test_df["date"] == date]

            # Check and close existing positions
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                if sym in data_dict and date in data_dict[sym].index:
                    current_price = data_dict[sym].loc[date, "close"]
                    ret = (current_price / pos.entry_price - 1) * pos.direction

                    # Stop-loss
                    if ret <= -STOP_LOSS_PCT:
                        pos.exit_date = date
                        pos.exit_price = current_price
                        pos.pnl = pos.size * ret - abs(pos.size) * TRANSACTION_COST_PCT
                        pos.pnl_pct = ret
                        pos.exit_reason = "stop_loss"
                        capital += pos.pnl
                        fold_trades.append(pos)
                        del open_positions[sym]
                        continue

                    # Take-profit
                    if ret >= TAKE_PROFIT_PCT:
                        pos.exit_date = date
                        pos.exit_price = current_price
                        pos.pnl = pos.size * ret - abs(pos.size) * TRANSACTION_COST_PCT
                        pos.pnl_pct = ret
                        pos.exit_reason = "take_profit"
                        capital += pos.pnl
                        fold_trades.append(pos)
                        del open_positions[sym]
                        continue

                    # Trailing stop
                    high_since = data_dict[sym].loc[:date, "high"].iloc[-5:].max()
                    trail_ret = (current_price / high_since - 1) if pos.direction == 1 else (
                        high_since / current_price - 1
                    )
                    if trail_ret <= -TRAILING_STOP_PCT and ret > 0:
                        pos.exit_date = date
                        pos.exit_price = current_price
                        pos.pnl = pos.size * ret - abs(pos.size) * TRANSACTION_COST_PCT
                        pos.pnl_pct = ret
                        pos.exit_reason = "trailing_stop"
                        capital += pos.pnl
                        fold_trades.append(pos)
                        del open_positions[sym]

            # Calculate current exposure
            total_exposure = sum(abs(p.size) for p in open_positions.values())
            exposure_pct = total_exposure / max(capital, 1)

            if exposure_pct >= MAX_PORTFOLIO_EXPOSURE:
                continue

            # Generate signals and open new positions
            for _, row in day_data.iterrows():
                sym = row["symbol"]
                if sym in open_positions:
                    continue

                pred_prob = row["pred_prob"]
                if pred_prob < 0.55 and pred_prob > 0.45:
                    continue  # Skip low-confidence predictions

                # Detect regime
                if sym in features_dict and date in features_dict[sym].index:
                    feat = features_dict[sym]
                    log_ret = feat.get("log_ret_1d", pd.Series(dtype=float))
                    regime = regime_detector.detect(log_ret).get(date, 0)
                    sig = combined_signal(feat.loc[:date].tail(1), regime)
                    signal_val = sig.iloc[-1] if len(sig) > 0 else 0
                else:
                    regime = 0
                    signal_val = 0

                # Determine direction
                if pred_prob >= 0.55:
                    direction = 1  # long
                elif pred_prob <= 0.45:
                    direction = -1  # short
                else:
                    continue

                # Confirm signal agrees with prediction
                if direction == 1 and signal_val < -0.3:
                    continue
                if direction == -1 and signal_val > 0.3:
                    continue

                # Volume confirmation: skip if volume is below average
                vol_ratio = row.get("volume_ratio", 1.0)
                if isinstance(vol_ratio, (int, float)) and vol_ratio < 0.8:
                    continue

                # Kelly position sizing
                drawdown = max(0, (peak_capital - capital) / peak_capital)
                win_prob = pred_prob if direction == 1 else (1 - pred_prob)
                kelly_frac = dynamic_kelly(win_prob, 1.0, drawdown)
                if kelly_frac <= 0:
                    continue

                position_size = capital * kelly_frac
                if position_size < 10:
                    continue

                # Check exposure limit
                if total_exposure + position_size > capital * MAX_PORTFOLIO_EXPOSURE:
                    position_size = max(0, capital * MAX_PORTFOLIO_EXPOSURE - total_exposure)
                    if position_size < 10:
                        continue

                entry_price = data_dict[sym].loc[date, "close"] if (
                    sym in data_dict and date in data_dict[sym].index
                ) else None
                if entry_price is None:
                    continue

                # Transaction cost on entry
                capital -= position_size * TRANSACTION_COST_PCT

                trade = Trade(
                    symbol=sym, entry_date=date, entry_price=entry_price,
                    size=position_size, direction=direction,
                    signal_strength=abs(signal_val), regime=regime,
                )
                open_positions[sym] = trade
                total_exposure += position_size

            # Record equity
            self.equity_curve.append({
                "date": date, "capital": capital, "fold": fold_idx,
                "open_positions": len(open_positions),
            })
            peak_capital = max(peak_capital, capital)

        # Close remaining positions at last available price
        for sym, pos in open_positions.items():
            if sym in data_dict and len(data_dict[sym]) > 0:
                last_price = data_dict[sym]["close"].iloc[-1]
                last_date = data_dict[sym].index[-1]
                ret = (last_price / pos.entry_price - 1) * pos.direction
                pos.exit_date = last_date
                pos.exit_price = last_price
                pos.pnl = pos.size * ret - abs(pos.size) * TRANSACTION_COST_PCT
                pos.pnl_pct = ret
                pos.exit_reason = "fold_end"
                capital += pos.pnl
                fold_trades.append(pos)

        self.trades.extend(fold_trades)

        wins = [t for t in fold_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in fold_trades)
        total_wagered = sum(t.size for t in fold_trades) if fold_trades else 1

        return {
            "fold": fold_idx,
            "n_trades": len(fold_trades),
            "wins": len(wins),
            "losses": len(fold_trades) - len(wins),
            "win_rate": len(wins) / max(len(fold_trades), 1) * 100,
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / max(total_wagered, 1) * 100, 2),
            "starting_capital": round(fold_start_capital, 2),
            "ending_capital": round(capital, 2),
        }

    def _compile_results(self, final_capital: float) -> dict:
        """Compile overall backtest results."""
        if not self.trades:
            return {"error": "No trades executed"}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_wagered = sum(t.size for t in self.trades)

        # Daily P&L for Sharpe calculation
        eq = pd.DataFrame(self.equity_curve)
        if len(eq) > 1:
            daily_returns = eq["capital"].pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)
                      if daily_returns.std() > 0 else 0)
            max_dd = self._max_drawdown(eq["capital"])
        else:
            sharpe = 0
            max_dd = 0

        gross_wins = sum(t.pnl for t in wins) if wins else 0
        gross_losses = abs(sum(t.pnl for t in losses)) if losses else 1

        # Average win/loss
        avg_win = gross_wins / max(len(wins), 1)
        avg_loss = gross_losses / max(len(losses), 1)

        results = {
            "initial_capital": INITIAL_CAPITAL,
            "final_capital": round(final_capital, 2),
            "total_return_pct": round((final_capital / INITIAL_CAPITAL - 1) * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / max(total_wagered, 1) * 100, 2),
            "n_trades": len(self.trades),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate": round(len(wins) / max(len(self.trades), 1) * 100, 1),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "profit_factor": round(gross_wins / max(gross_losses, 0.01), 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(avg_win / max(avg_loss, 0.01), 3),
            "n_folds": len(self.fold_results),
            "fold_results": self.fold_results,
        }
        return results

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return abs(dd.min()) if len(dd) > 0 else 0


class CPCVBacktester:
    """Combinatorial Purged Cross-Validation backtester."""

    def __init__(self, n_splits: int = CPCV_N_SPLITS,
                 n_test: int = CPCV_N_TEST_SPLITS,
                 purge_days: int = PURGE_DAYS):
        self.n_splits = n_splits
        self.n_test = n_test
        self.purge_days = purge_days

    def get_splits(self, dates: list) -> list[tuple[list, list]]:
        """Generate CPCV train/test splits."""
        n = len(dates)
        fold_size = n // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n
            folds.append(dates[start:end])

        splits = []
        for test_combo in combinations(range(self.n_splits), self.n_test):
            test_dates = []
            for idx in test_combo:
                test_dates.extend(folds[idx])

            train_dates = []
            for idx in range(self.n_splits):
                if idx not in test_combo:
                    fold_dates = folds[idx]
                    # Purge: remove dates near test boundaries
                    if self.purge_days > 0:
                        test_set = set(test_dates)
                        purged = []
                        for d in fold_dates:
                            too_close = False
                            for td in test_dates:
                                if abs((d - td).days) < self.purge_days:
                                    too_close = True
                                    break
                            if not too_close:
                                purged.append(d)
                        train_dates.extend(purged)
                    else:
                        train_dates.extend(fold_dates)

            if train_dates and test_dates:
                splits.append((sorted(train_dates), sorted(test_dates)))

        return splits

    def run(self, dataset: pd.DataFrame, features_dict: dict,
            data_dict: dict) -> dict:
        """Run CPCV backtest across all combinatorial splits."""
        dates = sorted(dataset["date"].unique())
        splits = self.get_splits(dates)
        print(f"CPCV: {len(splits)} splits from C({self.n_splits},{self.n_test})")

        all_results = []
        for i, (train_dates, test_dates) in enumerate(splits):
            train_mask = dataset["date"].isin(train_dates)
            test_mask = dataset["date"].isin(test_dates)
            train_df = dataset[train_mask]
            test_df = dataset[test_mask]

            if len(train_df) < 100 or len(test_df) < 50:
                continue

            model = EnsembleModel()
            try:
                model.fit(train_df, "target_up")
                probs = model.predict_proba(test_df)
                accuracy = ((probs >= 0.5) == test_df["target_up"].values).mean()
                all_results.append({
                    "split": i,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "accuracy": round(accuracy * 100, 1),
                })
                if (i + 1) % 3 == 0:
                    print(f"  Split {i+1}/{len(splits)}: acc={accuracy:.1%}")
            except Exception as e:
                print(f"  Split {i} failed: {e}")

        if not all_results:
            return {"error": "No valid CPCV splits"}

        accuracies = [r["accuracy"] for r in all_results]
        return {
            "n_splits": len(all_results),
            "mean_accuracy": round(np.mean(accuracies), 2),
            "std_accuracy": round(np.std(accuracies), 2),
            "min_accuracy": round(np.min(accuracies), 2),
            "max_accuracy": round(np.max(accuracies), 2),
            "split_results": all_results,
        }
