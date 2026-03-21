"""
Data Fetcher — Downloads OHLCV data for crypto assets via yfinance.
"""
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, LOOKBACK_YEARS, MIN_HISTORY_DAYS, OHLCV_INTERVAL


def fetch_ohlcv(
    symbols: list[str] | None = None,
    years: int = LOOKBACK_YEARS,
    interval: str = OHLCV_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for all assets. Returns {symbol: DataFrame}."""
    symbols = symbols or ASSETS
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, interval=interval, progress=False)
            if df.empty:
                print(f"  [WARN] No data for {sym}, skipping")
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "date"
            df = df.dropna()
            if len(df) < MIN_HISTORY_DAYS:
                print(f"  [WARN] {sym} only has {len(df)} days, need {MIN_HISTORY_DAYS}")
                continue
            data[sym] = df
            print(f"  [OK] {sym}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            print(f"  [ERR] {sym}: {e}")
    return data


def build_universe(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a multi-asset close-price DataFrame."""
    closes = {}
    for sym, df in data.items():
        closes[sym] = df["close"]
    universe = pd.DataFrame(closes)
    universe = universe.dropna(how="all").ffill()
    return universe


def fetch_and_build() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Convenience: fetch all data and build universe."""
    print("Fetching OHLCV data...")
    data = fetch_ohlcv()
    print(f"\nBuilding universe with {len(data)} assets...")
    universe = build_universe(data)
    print(f"Universe: {universe.shape[0]} days x {universe.shape[1]} assets")
    print(f"Date range: {universe.index[0].date()} to {universe.index[-1].date()}")
    return data, universe


if __name__ == "__main__":
    data, universe = fetch_and_build()
    os.makedirs("outputs", exist_ok=True)
    universe.to_csv("outputs/universe_closes.csv")
    print("\nSaved to outputs/universe_closes.csv")
