"""
Regime Detection — Identifies market regime (trending vs mean-reverting).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    REGIME_METHOD, REGIME_LOOKBACK,
    VOLATILITY_THRESHOLD_HIGH, VOLATILITY_THRESHOLD_LOW,
    HMM_N_STATES,
)


class RegimeDetector:
    """Detect market regime using volatility percentile or HMM."""

    TRENDING = 1
    MEAN_REVERTING = -1
    NEUTRAL = 0

    def __init__(self, method: str = REGIME_METHOD):
        self.method = method
        self.hmm_model = None

    def detect_volatility(self, returns: pd.Series) -> pd.Series:
        """Volatility-based regime detection.
        High vol = trending, low vol = mean-reverting.
        """
        vol = returns.rolling(20).std() * np.sqrt(365)
        vol_pct = vol.rolling(REGIME_LOOKBACK).rank(pct=True)

        regime = pd.Series(self.NEUTRAL, index=returns.index)
        regime[vol_pct >= VOLATILITY_THRESHOLD_HIGH] = self.TRENDING
        regime[vol_pct <= VOLATILITY_THRESHOLD_LOW] = self.MEAN_REVERTING
        return regime

    def detect_hmm(self, returns: pd.Series) -> pd.Series:
        """HMM-based regime detection."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            print("[WARN] hmmlearn not installed, falling back to volatility method")
            return self.detect_volatility(returns)

        clean = returns.dropna().values.reshape(-1, 1)
        if len(clean) < 100:
            return pd.Series(self.NEUTRAL, index=returns.index)

        model = GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(clean)
        states = model.predict(clean)

        # Map states to regime labels based on mean return
        state_means = {i: clean[states == i].mean() for i in range(HMM_N_STATES)}
        sorted_states = sorted(state_means, key=lambda x: state_means[x])

        regime_map = {}
        regime_map[sorted_states[0]] = self.MEAN_REVERTING  # lowest mean = bearish/MR
        regime_map[sorted_states[-1]] = self.TRENDING         # highest mean = bullish/trend
        for s in sorted_states[1:-1]:
            regime_map[s] = self.NEUTRAL

        mapped = pd.Series(
            [regime_map[s] for s in states],
            index=returns.dropna().index,
        )
        return mapped.reindex(returns.index, fill_value=self.NEUTRAL)

    def detect(self, returns: pd.Series) -> pd.Series:
        if self.method == "hmm":
            return self.detect_hmm(returns)
        return self.detect_volatility(returns)


def get_regime_weights(regime: int) -> tuple[float, float]:
    """Return (momentum_weight, mean_reversion_weight) for a given regime."""
    from config import REGIME_WEIGHT_MOMENTUM, REGIME_WEIGHT_MR, EQUAL_BLEND_WEIGHT

    if regime == RegimeDetector.TRENDING:
        return (REGIME_WEIGHT_MOMENTUM, 1 - REGIME_WEIGHT_MOMENTUM)
    elif regime == RegimeDetector.MEAN_REVERTING:
        return (1 - REGIME_WEIGHT_MR, REGIME_WEIGHT_MR)
    else:
        return (EQUAL_BLEND_WEIGHT, EQUAL_BLEND_WEIGHT)
