"""
Data ingestion layer.

Fetches and cleans market data from yfinance (free, no API key).
Maintains a clean price matrix for the entire universe.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import DataConfig

logger = logging.getLogger(__name__)


class DataFeed:
    """
    Fetches historical price data and computes derived series
    (returns, volume, etc.) for the trading universe.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.prices: Optional[pd.DataFrame] = None       # Adjusted close
        self.returns: Optional[pd.DataFrame] = None       # Simple returns
        self.log_returns: Optional[pd.DataFrame] = None   # Log returns
        self.volume: Optional[pd.DataFrame] = None
        self.high: Optional[pd.DataFrame] = None
        self.low: Optional[pd.DataFrame] = None
        self.open: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Download data for the full universe. Alias for fetch()."""
        return self.fetch()

    def fetch(self) -> None:
        """Download data for the full universe."""
        tickers = self.config.tickers
        years = self.config.years
        logger.info(f"Fetching data for {len(tickers)} assets, "
                     f"lookback={years} years")

        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=int(years * 365 * 1.1))

        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            raise ValueError("No data returned from yfinance")

        # Handle single-ticker edge case
        if len(tickers) == 1:
            self.prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            self.volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
            self.high = raw[["High"]].rename(columns={"High": tickers[0]})
            self.low = raw[["Low"]].rename(columns={"Low": tickers[0]})
            self.open = raw[["Open"]].rename(columns={"Open": tickers[0]})
        else:
            self.prices = raw["Close"]
            self.volume = raw["Volume"]
            self.high = raw["High"]
            self.low = raw["Low"]
            self.open = raw["Open"]

        # Drop assets with insufficient history
        min_obs = self.config.min_history_days
        valid_cols = self.prices.columns[self.prices.count() >= min_obs]
        dropped = set(self.prices.columns) - set(valid_cols)
        if dropped:
            logger.warning(f"Dropped assets with insufficient history: {dropped}")
        self.prices = self.prices[valid_cols]
        self.volume = self.volume[valid_cols]
        self.high = self.high[valid_cols]
        self.low = self.low[valid_cols]
        self.open = self.open[valid_cols]

        # Forward-fill then drop any remaining NaN rows at the start
        self.prices = self.prices.ffill().dropna(how="all")
        self.volume = self.volume.ffill().fillna(0)
        self.high = self.high.ffill().dropna(how="all")
        self.low = self.low.ffill().dropna(how="all")
        self.open = self.open.ffill().dropna(how="all")

        # Compute returns
        self.returns = self.prices.pct_change()
        self.log_returns = np.log(self.prices / self.prices.shift(1))

        logger.info(f"Data loaded: {self.prices.shape[0]} days, "
                     f"{self.prices.shape[1]} assets")

    def get_slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "DataFeed":
        """Return a DataFeed containing only data within [start, end]."""
        sliced = DataFeed(self.config)
        mask = (self.prices.index >= start) & (self.prices.index <= end)
        sliced.prices = self.prices.loc[mask].copy()
        sliced.returns = self.returns.loc[mask].copy()
        sliced.log_returns = self.log_returns.loc[mask].copy()
        sliced.volume = self.volume.loc[mask].copy()
        sliced.high = self.high.loc[mask].copy()
        sliced.low = self.low.loc[mask].copy()
        sliced.open = self.open.loc[mask].copy()
        return sliced

    @property
    def assets(self):
        return list(self.prices.columns)

    @property
    def dates(self):
        return self.prices.index
