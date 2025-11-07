"""Model training orchestration for MLB win probability models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None


@dataclass(slots=True)
class ModelResult:
    """Container storing fitted estimators and evaluation metrics."""

    name: str
    estimator: object
    auc: float
    feature_names: List[str]


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training and evaluation."""

    target_column: str = "home_win"
    group_column: str = "season"
    feature_columns: Optional[List[str]] = None
    calibration: bool = True
    n_splits: int = 5


class ModelTrainer:
    """Train interpretable and gradient boosting models."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()

    def _split(self, frame: pl.DataFrame) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        groups = frame[self.config.group_column].to_numpy()
        splitter = GroupKFold(n_splits=self.config.n_splits)
        for train_idx, test_idx in splitter.split(frame.to_numpy(), groups=groups, groups=groups):
            yield train_idx, test_idx

    def _features_and_target(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, List[str]]:
        feature_cols = self.config.feature_columns or [col for col in frame.columns if col not in {self.config.target_column, self.config.group_column}]
        X = frame.select(feature_cols).to_numpy()
        y = frame[self.config.target_column].to_numpy()
        return X, y, feature_cols

    def train_logistic(self, frame: pl.DataFrame) -> ModelResult:
        """Train a logistic regression with optional calibration."""

        X, y, features = self._features_and_target(frame)
        pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(max_iter=1000, penalty="l2")),
        ])
        if self.config.calibration:
            calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
            calibrated.fit(X, y)
            estimator = calibrated
            probs = calibrated.predict_proba(X)[:, 1]
        else:
            pipeline.fit(X, y)
            estimator = pipeline
            probs = pipeline.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, probs)
        return ModelResult(name="logistic_regression", estimator=estimator, auc=float(auc), feature_names=features)

    def train_lightgbm(self, frame: pl.DataFrame) -> ModelResult:
        """Train a gradient boosted tree model using LightGBM."""

        if lgb is None:
            raise RuntimeError("LightGBM is not installed")

        X, y, features = self._features_and_target(frame)
        dataset = lgb.Dataset(X, label=y, feature_name=features)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        }
        booster = lgb.train(params, dataset, num_boost_round=300)
        predictions = booster.predict(X)
        auc = roc_auc_score(y, predictions)
        return ModelResult(name="lightgbm", estimator=booster, auc=float(auc), feature_names=features)


__all__ = ["ModelTrainer", "TrainingConfig", "ModelResult"]
