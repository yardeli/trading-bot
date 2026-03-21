"""
Feature Engineering — Technical indicators, regime features, cross-asset signals.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RSI_PERIOD, RSI_SHORT_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    MOMENTUM_LOOKBACKS, VOLATILITY_WINDOW, ATR_PERIOD, VOLUME_MA_PERIOD,
    EMA_PERIODS, REGIME_LOOKBACK,
)


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger(series: pd.Series, period: int = BOLLINGER_PERIOD,
                      std: float = BOLLINGER_STD):
    sma = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    pct_b = (series - lower) / (upper - lower)
    z_score = (series - sma) / rolling_std.replace(0, np.nan)
    return sma, upper, lower, pct_b, z_score


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_returns(series: pd.Series, periods: list[int]) -> dict[str, pd.Series]:
    return {f"ret_{p}d": series.pct_change(p) for p in periods}


def compute_volatility(returns: pd.Series, window: int = VOLATILITY_WINDOW) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(365)


def compute_volume_features(volume: pd.Series, period: int = VOLUME_MA_PERIOD):
    vol_ma = volume.rolling(period).mean()
    vol_ratio = volume / vol_ma.replace(0, np.nan)
    return vol_ma, vol_ratio


def build_features_single(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Build features for a single asset OHLCV DataFrame."""
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Returns at multiple horizons
    for period in MOMENTUM_LOOKBACKS:
        feat[f"ret_{period}d"] = close.pct_change(period)

    # Log returns
    feat["log_ret_1d"] = np.log(close / close.shift(1))

    # RSI
    feat["rsi_14"] = compute_rsi(close, RSI_PERIOD)
    feat["rsi_6"] = compute_rsi(close, RSI_SHORT_PERIOD)

    # Bollinger Bands
    sma, upper, lower, pct_b, z_score = compute_bollinger(close)
    feat["bb_pct_b"] = pct_b
    feat["bb_zscore"] = z_score
    feat["bb_width"] = (upper - lower) / sma

    # ATR and volatility
    feat["atr"] = compute_atr(high, low, close)
    feat["atr_pct"] = feat["atr"] / close
    feat["volatility_20d"] = compute_volatility(feat["log_ret_1d"])

    # Volume features
    _, vol_ratio = compute_volume_features(volume)
    feat["volume_ratio"] = vol_ratio

    # EMAs and crossovers
    for p in EMA_PERIODS:
        feat[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
        feat[f"ema_{p}_dist"] = (close - feat[f"ema_{p}"]) / feat[f"ema_{p}"]
    feat["ema_cross_8_21"] = (feat["ema_8"] > feat["ema_21"]).astype(float)
    feat["ema_cross_21_50"] = (feat["ema_21"] > feat["ema_50"]).astype(float)

    # Momentum rank (within asset, percentile of recent returns)
    feat["mom_30d_rank"] = feat["ret_30d"].rolling(REGIME_LOOKBACK).rank(pct=True)

    # Volatility regime (rolling percentile)
    feat["vol_regime_pct"] = feat["volatility_20d"].rolling(REGIME_LOOKBACK).rank(pct=True)

    # Mean reversion signal strength
    feat["mr_signal"] = -feat["bb_zscore"]  # higher = more oversold
    feat["mr_rsi_signal"] = (50 - feat["rsi_14"]) / 50  # positive when oversold

    # Momentum signal
    feat["mom_signal"] = feat["ret_30d"].clip(-0.5, 0.5)

    # Drop raw EMA values (keep distances)
    for p in EMA_PERIODS:
        feat.drop(f"ema_{p}", axis=1, inplace=True)

    return feat


def build_features_universe(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build features for all assets."""
    features = {}
    for sym, df in data.items():
        feat = build_features_single(df, sym)
        features[sym] = feat
    return features


def build_cross_sectional_features(
    features: dict[str, pd.DataFrame], universe: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Add cross-sectional (relative) features."""
    # Cross-sectional momentum rank
    ret_30d = pd.DataFrame({
        sym: f["ret_30d"] for sym, f in features.items()
    })
    cs_rank = ret_30d.rank(axis=1, pct=True)

    # Cross-sectional volatility rank
    vol_20d = pd.DataFrame({
        sym: f["volatility_20d"] for sym, f in features.items()
    })
    vol_rank = vol_20d.rank(axis=1, pct=True)

    # BTC correlation (rolling)
    if "BTC-USD" in features:
        btc_ret = features["BTC-USD"]["log_ret_1d"]
        for sym, feat in features.items():
            feat["cs_mom_rank"] = cs_rank.get(sym, pd.Series(dtype=float))
            feat["cs_vol_rank"] = vol_rank.get(sym, pd.Series(dtype=float))
            if sym != "BTC-USD":
                asset_ret = feat["log_ret_1d"]
                aligned = pd.concat([btc_ret, asset_ret], axis=1).dropna()
                if len(aligned) > 30:
                    feat["btc_corr_30d"] = aligned.iloc[:, 0].rolling(30).corr(aligned.iloc[:, 1])
                else:
                    feat["btc_corr_30d"] = np.nan
            else:
                feat["btc_corr_30d"] = 1.0

    return features


def prepare_ml_dataset(
    features: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    forward_days: int = 7,
) -> pd.DataFrame:
    """Prepare ML dataset with features and forward return target."""
    rows = []
    for sym, feat in features.items():
        df = data[sym]
        combined = feat.copy()
        combined["symbol"] = sym
        # Forward return as target
        combined["target_ret"] = df["close"].pct_change(forward_days).shift(-forward_days)
        # Binary target: positive return?
        combined["target_up"] = (combined["target_ret"] > 0).astype(int)
        combined["date"] = combined.index
        rows.append(combined)

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.dropna(subset=["target_ret"])
    return dataset
