"""
Signal Generators — Momentum and Mean Reversion signal logic.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MR_RSI_OVERSOLD, MR_RSI_OVERBOUGHT, MR_ZSCORE_ENTRY, MR_ZSCORE_EXIT,
    MOM_LOOKBACK, MOM_THRESHOLD, MOM_TOP_N,
)


def mean_reversion_signal(features: pd.DataFrame) -> pd.Series:
    """Generate mean reversion signal [-1 to 1].
    Positive = buy (oversold), Negative = sell (overbought).
    """
    signal = pd.Series(0.0, index=features.index)

    rsi = features.get("rsi_14", pd.Series(dtype=float))
    zscore = features.get("bb_zscore", pd.Series(dtype=float))
    rsi_6 = features.get("rsi_6", pd.Series(dtype=float))

    # RSI-based signal
    rsi_signal = pd.Series(0.0, index=features.index)
    rsi_signal[rsi < MR_RSI_OVERSOLD] = 1.0
    rsi_signal[rsi > MR_RSI_OVERBOUGHT] = -1.0
    rsi_signal[(rsi >= MR_RSI_OVERSOLD) & (rsi <= MR_RSI_OVERBOUGHT)] = (
        (50 - rsi[(rsi >= MR_RSI_OVERSOLD) & (rsi <= MR_RSI_OVERBOUGHT)]) / 50
    )

    # Z-score based signal
    z_signal = pd.Series(0.0, index=features.index)
    z_signal[zscore <= MR_ZSCORE_ENTRY] = 1.0
    z_signal[zscore >= -MR_ZSCORE_ENTRY] = -1.0
    z_signal[(zscore > MR_ZSCORE_ENTRY) & (zscore < -MR_ZSCORE_ENTRY)] = -zscore[
        (zscore > MR_ZSCORE_ENTRY) & (zscore < -MR_ZSCORE_ENTRY)
    ] / abs(MR_ZSCORE_ENTRY)

    # Short RSI confirmation
    rsi6_signal = pd.Series(0.0, index=features.index)
    rsi6_signal[rsi_6 < 20] = 1.0
    rsi6_signal[rsi_6 > 80] = -1.0

    # Combine: weighted average
    signal = 0.4 * rsi_signal + 0.4 * z_signal + 0.2 * rsi6_signal
    return signal.clip(-1, 1)


def momentum_signal(features: pd.DataFrame) -> pd.Series:
    """Generate momentum signal [-1 to 1].
    Positive = bullish momentum, Negative = bearish.
    """
    ret_col = f"ret_{MOM_LOOKBACK}d"
    if ret_col not in features.columns:
        return pd.Series(0.0, index=features.index)

    ret = features[ret_col]
    signal = pd.Series(0.0, index=features.index)

    # Directional momentum
    signal[ret > MOM_THRESHOLD] = (ret[ret > MOM_THRESHOLD] / MOM_THRESHOLD).clip(0, 1)
    signal[ret < -MOM_THRESHOLD] = (ret[ret < -MOM_THRESHOLD] / MOM_THRESHOLD).clip(-1, 0)

    # EMA crossover confirmation
    ema_cross = features.get("ema_cross_8_21", pd.Series(1.0, index=features.index))
    signal = signal * (0.7 + 0.3 * ema_cross)

    return signal.clip(-1, 1)


def cross_sectional_momentum(
    features_dict: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    top_n: int = MOM_TOP_N,
) -> list[str]:
    """Select top N momentum assets on a given date."""
    scores = {}
    for sym, feat in features_dict.items():
        if date in feat.index:
            row = feat.loc[date]
            ret_30 = row.get("ret_30d", np.nan)
            if not np.isnan(ret_30):
                scores[sym] = ret_30

    if not scores:
        return []

    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:top_n]


def combined_signal(
    features: pd.DataFrame,
    regime: int,
) -> pd.Series:
    """Combine momentum and mean reversion signals based on regime."""
    from strategy.regime import get_regime_weights

    mom_w, mr_w = get_regime_weights(regime)
    mom = momentum_signal(features)
    mr = mean_reversion_signal(features)
    return (mom_w * mom + mr_w * mr).clip(-1, 1)
