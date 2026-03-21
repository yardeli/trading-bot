"""
Signal Aggregation / Ensemble Layer.

Combines signals from multiple alpha models into a single
composite signal using various ensemble methods.

Hedge Fund Usage:
    - Most multi-strategy funds run 5-20 alpha models in parallel.
    - Signals are combined using inverse-variance weighting, Bayesian
      model averaging, or meta-learning (stacking).
    - Model weights are updated based on recent performance (adaptive).
"""
import numpy as np
import pandas as pd


class SignalAggregator:
    """
    Combines multiple alpha signals into a single composite signal.

    Methods:
        equal_weight: Simple average of all signals.
        inverse_vol: Weight by inverse of signal volatility (Sharpe-like).
        performance_weighted: Weight by recent realized IC (information coefficient).
    """

    def __init__(self, method: str = "inverse_vol", ic_lookback: int = 63):
        self.method = method
        self.ic_lookback = ic_lookback
        self.model_weights: dict[str, float] = {}

    def aggregate(
        self,
        signals: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine multiple signal DataFrames into one.

        Args:
            signals: Dict mapping model_name -> signal DataFrame.
            returns: Realized returns DataFrame (for performance weighting).

        Returns:
            Combined signal DataFrame.
        """
        if not signals:
            raise ValueError("No signals to aggregate")

        if len(signals) == 1:
            name, sig = next(iter(signals.items()))
            self.model_weights = {name: 1.0}
            return sig

        if self.method == "equal_weight":
            return self._equal_weight(signals)
        elif self.method == "inverse_vol":
            return self._inverse_vol(signals, returns)
        elif self.method == "performance_weighted":
            return self._performance_weighted(signals, returns)
        else:
            return self._equal_weight(signals)

    def _equal_weight(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simple average of all signals."""
        combined = None
        for name, sig in signals.items():
            if combined is None:
                combined = sig.copy()
            else:
                combined = combined.add(sig, fill_value=0)
            self.model_weights[name] = 1.0 / len(signals)

        return combined / len(signals)

    def _inverse_vol(
        self, signals: dict[str, pd.DataFrame], returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Weight each model by the inverse of its signal volatility.
        Models with more stable signals get higher weight.
        This is analogous to risk parity at the signal level.
        """
        vols = {}
        for name, sig in signals.items():
            # Compute the volatility of each model's signal changes
            sig_diff = sig.diff()
            vols[name] = sig_diff.std().mean() + 1e-8

        # Inverse vol weights
        total_inv_vol = sum(1.0 / v for v in vols.values())
        weights = {name: (1.0 / v) / total_inv_vol for name, v in vols.items()}
        self.model_weights = weights

        combined = None
        for name, sig in signals.items():
            weighted = sig * weights[name]
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)

        return combined

    def _performance_weighted(
        self, signals: dict[str, pd.DataFrame], returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Weight models by their recent Information Coefficient (IC).
        IC = rank correlation between signal and subsequent returns.

        Models that have been predicting well recently get higher weight.
        This is adaptive — it automatically shifts capital toward
        whatever is working in the current regime.
        """
        ics = {}
        for name, sig in signals.items():
            # Compute rolling IC: rank correlation of signal vs next-day returns
            rolling_ic = self._compute_rolling_ic(sig, returns)
            # Use mean IC over lookback as the weight
            recent_ic = rolling_ic.iloc[-self.ic_lookback:].mean()
            ics[name] = max(recent_ic, 0.0)  # Don't give negative weight

        total_ic = sum(ics.values())
        if total_ic < 1e-8:
            # All models are bad — fall back to equal weight
            return self._equal_weight(signals)

        weights = {name: ic / total_ic for name, ic in ics.items()}
        self.model_weights = weights

        combined = None
        for name, sig in signals.items():
            weighted = sig * weights[name]
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)

        return combined

    @staticmethod
    def _compute_rolling_ic(
        signals: pd.DataFrame, returns: pd.DataFrame, window: int = 21
    ) -> pd.Series:
        """
        Compute rolling Information Coefficient.
        IC = Spearman rank correlation between signal and next-period returns.

        Only computes over the last `max_compute` rows for performance.
        """
        from scipy.stats import spearmanr

        # Only compute IC over the most recent portion to avoid O(n) blowup
        max_compute = 126  # ~6 months is plenty for IC estimation
        n = len(signals)
        compute_start = max(window, n - max_compute)

        shifted_sig = signals.shift(1)
        ics = []

        for i in range(compute_start, n):
            sig_row = shifted_sig.iloc[i]
            ret_idx = signals.index[i]
            if ret_idx not in returns.index:
                ics.append(0.0)
                continue
            ret_row = returns.loc[ret_idx]
            valid = sig_row.notna() & ret_row.notna()
            if valid.sum() < 5:
                ics.append(0.0)
                continue

            corr, _ = spearmanr(sig_row[valid], ret_row[valid])
            ics.append(corr if not np.isnan(corr) else 0.0)

        # Pad the beginning
        ic_series = pd.Series(
            [0.0] * compute_start + ics, index=signals.index
        )
        return ic_series
