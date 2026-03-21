"""
ML Ensemble — XGBoost + LightGBM + Ridge for return prediction.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import XGB_PARAMS, LGB_PARAMS, ENSEMBLE_WEIGHTS

FEATURE_COLS = [
    "ret_7d", "ret_14d", "ret_30d", "ret_60d", "ret_90d",
    "rsi_14", "rsi_6", "bb_pct_b", "bb_zscore", "bb_width",
    "atr_pct", "volatility_20d", "volume_ratio",
    "ema_8_dist", "ema_21_dist", "ema_50_dist", "ema_200_dist",
    "ema_cross_8_21", "ema_cross_21_50",
    "mom_30d_rank", "vol_regime_pct",
    "mr_signal", "mr_rsi_signal", "mom_signal",
    "cs_mom_rank", "cs_vol_rank", "btc_corr_30d",
]


class EnsembleModel:
    """Ensemble of XGBoost, LightGBM, and Ridge for return prediction."""

    def __init__(self, weights: list[float] | None = None):
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].fillna(0).values
        return X, available

    def fit(self, train_df: pd.DataFrame, target_col: str = "target_up"):
        import xgboost as xgb
        import lightgbm as lgb

        X, used_cols = self._get_features(train_df)
        y = train_df[target_col].values
        self.feature_cols_used = used_cols

        # Recency-weighted samples: exponential decay, recent data 3x more important
        if "date" in train_df.columns:
            dates = pd.to_datetime(train_df["date"])
            days_ago = (dates.max() - dates).dt.days.values.astype(float)
            sample_weight = np.exp(-days_ago / (len(days_ago) * 0.5))
            sample_weight = sample_weight / sample_weight.mean()  # normalize
        else:
            sample_weight = np.ones(len(y))

        # Scale for Ridge
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS, eval_metric="logloss")
        xgb_model.fit(X_scaled, y, sample_weight=sample_weight)
        self.models["xgboost"] = xgb_model

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(X_scaled, y, sample_weight=sample_weight)
        self.models["lightgbm"] = lgb_model

        # Ridge (probability via sigmoid of decision function)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_scaled, y, sample_weight=sample_weight)
        self.models["ridge"] = ridge_model

        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive return."""
        X = df[self.feature_cols_used].fillna(0).values
        X_scaled = self.scaler.transform(X)

        probs = np.zeros(len(X_scaled))

        # XGBoost
        xgb_prob = self.models["xgboost"].predict_proba(X_scaled)[:, 1]
        probs += self.weights[0] * xgb_prob

        # LightGBM
        lgb_prob = self.models["lightgbm"].predict_proba(X_scaled)[:, 1]
        probs += self.weights[1] * lgb_prob

        # Ridge (sigmoid transform)
        ridge_raw = self.models["ridge"].predict(X_scaled)
        ridge_prob = 1 / (1 + np.exp(-ridge_raw))
        probs += self.weights[2] * ridge_prob

        return probs

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree models."""
        importances = {}
        for name in ["xgboost", "lightgbm"]:
            model = self.models.get(name)
            if model and hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
        if not importances:
            return pd.DataFrame()
        imp_df = pd.DataFrame(importances, index=self.feature_cols_used)
        imp_df["mean"] = imp_df.mean(axis=1)
        return imp_df.sort_values("mean", ascending=False)
