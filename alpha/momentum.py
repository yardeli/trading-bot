"""
Momentum Alpha Models.

Implements time-series and cross-sectional momentum strategies
as described in Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
and Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers".

Mathematical Intuition:
    - Time-series momentum: assets that have gone up tend to keep going up
      (and vice versa). Signal = sign(past return) * vol-scaled position.
    - Cross-sectional momentum: long top decile, short bottom decile of
      past returns. Captures relative outperformance.

When it works: Trending markets, macro regime shifts, slow information diffusion.
When it fails: Momentum crashes (sharp reversals), choppy/mean-reverting markets.
"""
import numpy as np
import pandas as pd

from alpha.base import AlphaModel
from config import AlphaConfig
from data.feed import DataFeed


class TimeSeriesMomentum(AlphaModel):
    """
    Moskowitz-style time-series momentum.

    For each asset independently:
        signal_i = sign(r_{t-slow, t}) * (vol_target / realized_vol_i)

    This is the "trend following" signal used by CTAs and managed futures funds.
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="ts_momentum")

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data.prices
        returns = data.returns

        # Compute lookback returns
        fast_ret = prices.pct_change(self.config.momentum_fast)
        slow_ret = prices.pct_change(self.config.momentum_slow)

        # Blend fast and slow signals using continuous returns (not sign)
        # Normalize by rolling std to get z-score-like signals
        fast_vol = returns.rolling(self.config.momentum_fast).std() * np.sqrt(self.config.momentum_fast)
        slow_vol = returns.rolling(self.config.momentum_slow).std() * np.sqrt(self.config.momentum_slow)
        fast_z = fast_ret / (fast_vol + 1e-8)
        slow_z = slow_ret / (slow_vol + 1e-8)
        blended = 0.5 * fast_z.clip(-2, 2) + 0.5 * slow_z.clip(-2, 2)

        # Volatility scaling: target a specific vol per asset
        realized_vol = returns.rolling(63).std() * np.sqrt(252)
        vol_scalar = self.config.momentum_vol_target / (realized_vol + 1e-8)
        vol_scalar = vol_scalar.clip(0.1, 3.0)  # Safety bounds

        signals = blended * vol_scalar

        return self._clip_signals(signals)


class CrossSectionalMomentum(AlphaModel):
    """
    Jegadeesh-Titman cross-sectional momentum.

    Rank assets by past 12-month return (skipping most recent month
    to avoid short-term reversal). Go long top quintile, short bottom quintile.

    The skip-month is critical — without it, the strategy picks up
    short-term reversal instead of momentum.
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="xs_momentum")

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data.prices

        # 12-1 momentum: 12-month return, skip most recent month
        ret_12m = prices.pct_change(252)
        ret_1m = prices.pct_change(21)
        momentum_signal = ret_12m - ret_1m  # 12-1 momentum

        # Cross-sectional rank normalization
        signals = self._rank_normalize(momentum_signal)

        return self._clip_signals(signals)


class MomentumWithVolBreak(AlphaModel):
    """
    Enhanced momentum that reduces exposure during volatility spikes.

    Intuition: momentum strategies crash during volatility regime changes.
    By scaling down when vol is elevated, we avoid the worst drawdowns.

    This is how most sophisticated CTAs manage their trend-following books.
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="momentum_vol_break")
        self._ts_mom = TimeSeriesMomentum(config)

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Base momentum signal
        base_signal = self._ts_mom.generate_signals(data, features)

        # Compute volatility regime indicator
        returns = data.returns
        short_vol = returns.rolling(10).std() * np.sqrt(252)
        long_vol = returns.rolling(63).std() * np.sqrt(252)

        # Vol ratio > 1 means vol is elevated relative to recent history
        vol_ratio = short_vol / (long_vol + 1e-8)

        # Scale down when vol is elevated (smooth scaling, not binary)
        vol_brake = (1.0 / vol_ratio).clip(0.2, 1.0)

        # Correlation crash filter: measure average cross-asset correlation
        # When correlations spike, momentum tends to crash
        rolling_corr = returns.rolling(21).corr()
        # Get mean pairwise correlation per date
        n_assets = len(returns.columns)
        if n_assets > 1:
            avg_corr = rolling_corr.groupby(level=0).apply(
                lambda x: (x.values.sum() - n_assets) / (n_assets * (n_assets - 1))
                if x.shape[0] == n_assets else 0.5
            )
            # Normal correlation ~0.3, crisis ~0.7+
            # Scale down when correlation is abnormally high
            corr_brake = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
            for col in returns.columns:
                corr_brake[col] = (0.6 / (avg_corr + 1e-8)).clip(0.3, 1.0)
        else:
            corr_brake = 1.0

        signals = base_signal * vol_brake * corr_brake

        return self._clip_signals(signals)
