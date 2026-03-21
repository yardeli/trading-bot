"""
Feature engineering engine.

Generates a rich set of predictive features from raw market data.
Every feature is computed as a cross-sectional z-score (rank-normalized)
to ensure stationarity and comparability across assets.
"""
import numpy as np
import pandas as pd

from config import FeatureConfig
from data.feed import DataFeed


class FeatureEngine:
    """
    Produces a 3D feature tensor: (dates × assets × features).
    Stored as a dict of DataFrames, keyed by feature name.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.features: dict[str, pd.DataFrame] = {}

    def generate(self, data: DataFeed) -> dict[str, pd.DataFrame]:
        """Compute all features from the data feed."""
        self.features = {}
        prices = data.prices
        returns = data.returns
        volume = data.volume
        high = data.high
        low = data.low

        # --- Momentum features ---
        for w in self.config.momentum_windows:
            # Price momentum (total return over window)
            mom = prices.pct_change(w)
            self.features[f"mom_{w}d"] = self._cross_sectional_zscore(mom)

            # Risk-adjusted momentum (Sharpe-like)
            roll_ret = returns.rolling(w).mean()
            roll_vol = returns.rolling(w).std()
            sharpe = roll_ret / (roll_vol + 1e-8)
            self.features[f"sharpe_{w}d"] = self._cross_sectional_zscore(sharpe)

        # --- Volatility features ---
        for w in self.config.vol_windows:
            # Realized volatility (annualized)
            rvol = returns.rolling(w).std() * np.sqrt(252)
            self.features[f"rvol_{w}d"] = self._cross_sectional_zscore(rvol)

            # Volatility of volatility
            vol_of_vol = rvol.rolling(w).std()
            self.features[f"vov_{w}d"] = self._cross_sectional_zscore(vol_of_vol)

        # --- Parkinson volatility (uses high/low) ---
        log_hl = np.log(high / (low + 1e-8))
        parkinson = log_hl.rolling(21).apply(
            lambda x: np.sqrt((1 / (4 * len(x) * np.log(2))) * (x ** 2).sum()) * np.sqrt(252),
            raw=True
        )
        self.features["parkinson_vol"] = self._cross_sectional_zscore(parkinson)

        # --- Mean reversion features ---
        for w in self.config.mean_rev_windows:
            # Distance from moving average (z-score)
            ma = prices.rolling(w).mean()
            ma_std = prices.rolling(w).std()
            z = (prices - ma) / (ma_std + 1e-8)
            self.features[f"ma_zscore_{w}d"] = self._cross_sectional_zscore(z)

        # --- RSI ---
        rsi = self._compute_rsi(prices, self.config.rsi_period)
        # Transform RSI to [-1, 1] range centered at 50
        self.features["rsi"] = (rsi - 50) / 50

        # --- Bollinger Band position ---
        bb_ma = prices.rolling(self.config.bb_window).mean()
        bb_std = prices.rolling(self.config.bb_window).std()
        bb_pos = (prices - bb_ma) / (self.config.bb_std * bb_std + 1e-8)
        self.features["bb_position"] = bb_pos

        # --- Volume features ---
        # Relative volume (current vs average)
        for w in [5, 21]:
            avg_vol = volume.rolling(w).mean()
            rel_vol = volume / (avg_vol + 1e-8)
            self.features[f"rel_volume_{w}d"] = self._cross_sectional_zscore(rel_vol)

        # Volume-weighted price trend
        vwap_proxy = (returns * volume).rolling(21).sum() / (volume.rolling(21).sum() + 1e-8)
        self.features["vwap_trend"] = self._cross_sectional_zscore(vwap_proxy)

        # --- Cross-asset features ---
        # Beta to SPY (if SPY is in the universe)
        if "SPY" in returns.columns:
            spy_ret = returns["SPY"]
            for col in returns.columns:
                if col == "SPY":
                    continue
                cov = returns[col].rolling(63).cov(spy_ret)
                var = spy_ret.rolling(63).var()
                beta = cov / (var + 1e-8)
                if f"beta_spy" not in self.features:
                    self.features["beta_spy"] = pd.DataFrame(index=returns.index, columns=returns.columns)
                self.features["beta_spy"][col] = beta
            self.features["beta_spy"]["SPY"] = 1.0

        # --- Autocorrelation ---
        ac1 = returns.rolling(21).apply(lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False)
        self.features["autocorr_1d"] = ac1

        # --- Skewness and Kurtosis ---
        skew = returns.rolling(63).skew()
        self.features["skew_63d"] = self._cross_sectional_zscore(skew)
        kurt = returns.rolling(63).apply(lambda x: x.kurtosis() if len(x) > 3 else 0, raw=False)
        self.features["kurt_63d"] = self._cross_sectional_zscore(kurt)

        # --- Max drawdown (rolling) ---
        rolling_max = prices.rolling(63).max()
        drawdown = (prices - rolling_max) / (rolling_max + 1e-8)
        self.features["drawdown_63d"] = drawdown

        return self.features

    def get_feature_matrix(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        For a given date, return a (assets × features) DataFrame.
        Useful for ML models that need a cross-sectional feature matrix.
        """
        rows = {}
        for fname, fdf in self.features.items():
            if date in fdf.index:
                rows[fname] = fdf.loc[date]
        return pd.DataFrame(rows)

    def get_feature_panel(self) -> pd.DataFrame:
        """
        Return a long-format panel: (date, asset, feature1, feature2, ...).
        Used for ML training.
        """
        panels = []
        for fname, fdf in self.features.items():
            stacked = fdf.stack().rename(fname)
            panels.append(stacked)

        if not panels:
            return pd.DataFrame()

        panel = pd.concat(panels, axis=1)
        panel.index.names = ["date", "asset"]
        return panel

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize each row (date) to zero mean, unit variance across assets."""
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1)
        return df.sub(row_mean, axis=0).div(row_std + 1e-8, axis=0)

    @staticmethod
    def _compute_rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Wilder's RSI."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))
