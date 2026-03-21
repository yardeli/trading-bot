"""
Base class for alpha models.

Every alpha model takes market data and features, and outputs
a signal DataFrame: (dates × assets) with values in [-1, 1].

Positive = long signal, negative = short signal.
Magnitude = conviction.
"""
from abc import ABC, abstractmethod

import pandas as pd

from config import AlphaConfig
from data.feed import DataFeed


class AlphaModel(ABC):
    """Base alpha model interface."""

    def __init__(self, config: AlphaConfig, name: str = "base"):
        self.config = config
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        data: DataFeed,
        features: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate trading signals.

        Returns:
            DataFrame of shape (dates × assets), values in [-1, 1].
        """
        pass

    def _clip_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Clip signals to [-1, 1] range."""
        return signals.clip(-1, 1)

    def _rank_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank normalization to [-1, 1]."""
        ranked = df.rank(axis=1, pct=True)
        return 2 * ranked - 1
