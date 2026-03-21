"""
Machine Learning Alpha Model.

Uses gradient boosting (XGBoost-style via sklearn) to predict
forward returns from the feature matrix.

Mathematical Intuition:
    Ensemble of decision trees, each fit on the residuals of the previous.
    The model learns nonlinear relationships between features and future returns.
    L2 regularization + shallow depth + min_samples_leaf prevent overfitting.

When it works: When features have genuine nonlinear predictive power.
When it fails: Regime changes that invalidate learned patterns, overfitting
               to noise (mitigated by strict regularization and walk-forward testing).

Hedge Fund Usage:
    - Most quant funds use gradient boosting as one of several alpha models.
    - Features are typically cross-sectional z-scores (as we produce).
    - Walk-forward retraining prevents look-ahead bias.
    - Feature importance is monitored for model decay detection.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from alpha.base import AlphaModel
from config import AlphaConfig
from data.feed import DataFeed

logger = logging.getLogger(__name__)


class MLAlpha(AlphaModel):
    """
    Gradient Boosting alpha model with walk-forward retraining.

    Predicts N-day forward returns from the cross-sectional feature matrix.
    Outputs are rank-normalized to produce signals in [-1, 1].
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="ml_alpha")
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance: dict[str, float] = {}
        self.last_train_date = None

    def generate_signals(
        self,
        data: DataFeed,
        features: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        prices = data.prices
        returns = data.returns
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Build the panel dataset: (date, asset) -> features
        panel = self._build_panel(features, prices)
        if panel.empty:
            return signals

        # Compute forward returns (the target variable)
        horizon = self.config.ml_forward_return_horizon
        fwd_ret = prices.pct_change(horizon).shift(-horizon)
        fwd_ret_stacked = fwd_ret.stack()
        fwd_ret_stacked.name = "fwd_return"
        fwd_ret_stacked.index.names = ["date", "asset"]

        # Join features with forward returns
        panel = panel.join(fwd_ret_stacked, how="inner")
        panel = panel.replace([np.inf, -np.inf], np.nan).dropna()

        if len(panel) < 500:
            logger.warning("Insufficient data for ML training")
            return signals

        feature_cols = [c for c in panel.columns if c != "fwd_return"]
        dates = panel.index.get_level_values("date").unique().sort_values()

        train_window = self.config.ml_train_window
        # Scale retrain frequency with universe size to prevent hanging
        n_assets = len(prices.columns)
        retrain_freq = max(self.config.ml_retrain_frequency, n_assets * 5)

        # Reduce model complexity for large universes
        n_estimators = self.config.ml_n_estimators
        if n_assets > 10:
            n_estimators = min(n_estimators, 100)
        if n_assets > 20:
            n_estimators = min(n_estimators, 50)

        # Walk-forward: train on past, predict on present
        for i, date in enumerate(dates):
            if i < train_window:
                continue

            # Retrain periodically
            if self.model is None or (i % retrain_freq == 0):
                train_start_idx = max(0, i - train_window)
                train_dates = dates[train_start_idx:i]
                train_mask = panel.index.get_level_values("date").isin(train_dates)
                train_data = panel[train_mask]

                if len(train_data) < 200:
                    continue

                X_train = train_data[feature_cols].values
                y_train = train_data["fwd_return"].values

                # Winsorize targets to reduce outlier influence
                y_clip = np.clip(y_train, np.percentile(y_train, 1), np.percentile(y_train, 99))

                self.scaler.fit(X_train)
                X_scaled = self.scaler.transform(X_train)

                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=self.config.ml_max_depth,
                    min_samples_leaf=self.config.ml_min_samples_leaf,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
                self.model.fit(X_scaled, y_clip)

                # Track feature importance
                self.feature_importance = dict(
                    zip(feature_cols, self.model.feature_importances_)
                )
                self.last_train_date = date
                logger.debug(f"ML model retrained at {date}, "
                             f"train size={len(train_data)}")

            # Predict for current date
            if self.model is None:
                continue

            current_mask = panel.index.get_level_values("date") == date
            current_data = panel[current_mask]
            if current_data.empty:
                continue

            X_pred = self.scaler.transform(current_data[feature_cols].values)
            predictions = self.model.predict(X_pred)

            # Store predictions for each asset
            for asset, pred in zip(
                current_data.index.get_level_values("asset"), predictions
            ):
                signals.loc[date, asset] = pred

        # Rank-normalize signals cross-sectionally
        signals = self._rank_normalize(signals)
        return self._clip_signals(signals)

    def _build_panel(
        self, features: dict[str, pd.DataFrame], prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert feature dict into a stacked panel DataFrame."""
        panels = []
        for fname, fdf in features.items():
            # Align with prices
            aligned = fdf.reindex(index=prices.index, columns=prices.columns)
            stacked = aligned.stack()
            stacked.name = fname
            panels.append(stacked)

        if not panels:
            return pd.DataFrame()

        panel = pd.concat(panels, axis=1)
        panel.index.names = ["date", "asset"]
        return panel.replace([np.inf, -np.inf], np.nan).dropna()

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances from the last trained model."""
        return dict(
            sorted(self.feature_importance.items(), key=lambda x: -x[1])
        )
