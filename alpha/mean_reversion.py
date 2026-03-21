"""
Mean Reversion & Statistical Arbitrage Alpha Models.

Implements Ornstein-Uhlenbeck mean reversion and cointegration-based
pairs/basket trading.

Mathematical Intuition:
    The Ornstein-Uhlenbeck process: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    where theta = speed of mean reversion, mu = long-run mean.
    Half-life = ln(2) / theta.

    If the half-life is short (< 21 days), the spread mean-reverts quickly
    enough to trade profitably after transaction costs.

When it works: Range-bound markets, liquid pairs with fundamental linkages.
When it fails: Structural breaks, trending markets, regime changes that
               break historical relationships.
"""
import numpy as np
import pandas as pd
from scipy import stats

from alpha.base import AlphaModel
from config import AlphaConfig
from data.feed import DataFeed


class OUMeanReversion(AlphaModel):
    """
    Single-asset mean reversion using z-score of price relative to a
    moving average, calibrated as an Ornstein-Uhlenbeck process.

    Estimates the half-life of mean reversion for each asset.
    Only trades assets where the estimated half-life is between
    2 and the configured max, indicating genuine mean reversion
    (not noise and not too slow).
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="ou_mean_reversion")

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data.prices
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < self.config.mean_rev_halflife * 3:
                continue

            # Estimate OU half-life using regression:
            # delta_x = alpha + beta * x_{t-1} + epsilon
            # half_life = -ln(2) / beta
            # Use 126-day lookback (more adaptive to regime changes)
            lookback = min(len(series), 126)
            y = series.diff().iloc[-lookback:]
            x = series.shift(1).iloc[-lookback:]
            valid = y.notna() & x.notna()
            y, x = y[valid], x[valid]

            if len(y) < 30:
                continue

            slope, intercept, _, p_value, _ = stats.linregress(x, y)

            # Only trade if mean reversion is statistically significant
            if slope >= 0 or p_value > 0.10:
                continue  # Not mean-reverting

            half_life = -np.log(2) / slope
            if half_life < 2 or half_life > self.config.mean_rev_halflife * 3:
                continue  # Too fast (noise) or too slow (can't trade)

            # Compute z-score using estimated half-life as the lookback
            hl_int = max(int(half_life), 2)
            ma = prices[col].rolling(hl_int).mean()
            std = prices[col].rolling(hl_int).std()
            z = (prices[col] - ma) / (std + 1e-8)

            # Generate signals: short when z > entry, long when z < -entry
            signal = pd.Series(0.0, index=prices.index)
            signal[z > self.config.mean_rev_entry_z] = -1.0
            signal[z < -self.config.mean_rev_entry_z] = 1.0
            # Reduce signal as z approaches exit threshold
            exit_mask = z.abs() < self.config.mean_rev_exit_z
            signal[exit_mask] = 0.0
            # Scale by z-score magnitude (stronger signal for larger deviations)
            signal = signal * (z.abs() / self.config.mean_rev_entry_z).clip(0, 1)
            # Flip sign for the signal (buy low, sell high)
            signal = -signal  # Negative z = buy, positive z = sell

            signals[col] = signal

        return self._clip_signals(signals)


class PairsTrading(AlphaModel):
    """
    Cointegration-based pairs trading.

    Uses the Engle-Granger two-step method:
    1. Identify cointegrated pairs using ADF test on the spread.
    2. Trade the spread when it deviates from its mean.

    The hedge ratio is estimated using OLS regression.
    Pairs are re-evaluated periodically (formation period).
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="pairs_trading")

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data.prices
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Find cointegrated pairs
        assets = prices.columns.tolist()
        n = len(assets)
        if n < 2:
            return signals

        # Test all pairs for cointegration using a rolling window
        formation = self.config.stat_arb_formation_period
        pairs = []

        # Use the last formation_period of data for pair identification
        recent_prices = prices.iloc[-formation:]

        for i in range(min(n, 15)):  # Limit pairs search for performance
            for j in range(i + 1, min(n, 15)):
                a, b = assets[i], assets[j]
                pa = recent_prices[a].dropna()
                pb = recent_prices[b].dropna()
                common_idx = pa.index.intersection(pb.index)
                if len(common_idx) < formation // 2:
                    continue
                pa, pb = pa[common_idx], pb[common_idx]

                # OLS hedge ratio: a = beta * b + alpha + epsilon
                slope, intercept, r_value, _, _ = stats.linregress(pb, pa)
                spread = pa - slope * pb - intercept

                # ADF test on the spread (simplified: check if spread is stationary)
                # Use the half-life test as a proxy
                spread_diff = spread.diff().dropna()
                spread_lag = spread.shift(1).dropna()
                common = spread_diff.index.intersection(spread_lag.index)
                if len(common) < 20:
                    continue

                s_slope, _, _, s_pval, _ = stats.linregress(
                    spread_lag[common], spread_diff[common]
                )

                if s_slope >= 0 or s_pval > 0.10:
                    continue  # Spread is not mean-reverting

                half_life = -np.log(2) / s_slope
                if half_life < 1 or half_life > 63:
                    continue

                pairs.append((a, b, slope, intercept, half_life, r_value ** 2))

        # Sort by R-squared (best cointegration first), take top pairs
        pairs.sort(key=lambda x: -x[5])
        top_pairs = pairs[:5]

        # Generate signals for each pair
        for a, b, hedge_ratio, intercept, half_life, _ in top_pairs:
            spread = prices[a] - hedge_ratio * prices[b] - intercept
            hl_int = max(int(half_life), 2)
            spread_ma = spread.rolling(hl_int).mean()
            spread_std = spread.rolling(hl_int).std()
            z = (spread - spread_ma) / (spread_std + 1e-8)

            # Entry signals
            entry_z = self.config.stat_arb_entry_z
            exit_z = self.config.stat_arb_exit_z
            stop_z = self.config.stat_arb_stop_z

            pair_signal_a = pd.Series(0.0, index=prices.index)
            pair_signal_b = pd.Series(0.0, index=prices.index)

            # Short spread when z > entry_z (short a, long b)
            mask_short = z > entry_z
            pair_signal_a[mask_short] = -1.0
            pair_signal_b[mask_short] = hedge_ratio

            # Long spread when z < -entry_z (long a, short b)
            mask_long = z < -entry_z
            pair_signal_a[mask_long] = 1.0
            pair_signal_b[mask_long] = -hedge_ratio

            # Exit when spread mean-reverts
            mask_exit = z.abs() < exit_z
            pair_signal_a[mask_exit] = 0.0
            pair_signal_b[mask_exit] = 0.0

            # Stop-loss
            mask_stop = z.abs() > stop_z
            pair_signal_a[mask_stop] = 0.0
            pair_signal_b[mask_stop] = 0.0

            # Scale by number of pairs to avoid over-concentration
            scale = 1.0 / max(len(top_pairs), 1)
            signals[a] += pair_signal_a * scale
            signals[b] += pair_signal_b * scale

        return self._clip_signals(signals)
