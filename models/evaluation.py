"""Evaluation helpers for model outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, balanced_accuracy_score, roc_auc_score


@dataclass(slots=True)
class MetricBundle:
    """Standard evaluation metrics for binary classifiers."""

    auc: float
    brier: float
    logloss: float
    balanced_accuracy: float


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricBundle:
    """Compute core evaluation metrics."""

    predictions = (y_prob >= 0.5).astype(int)
    return MetricBundle(
        auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        logloss=float(log_loss(y_true, y_prob)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
    )


__all__ = ["compute_metrics", "MetricBundle"]
