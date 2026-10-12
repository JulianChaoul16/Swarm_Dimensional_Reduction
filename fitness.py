"""ML evaluation and feature-count penalty for feature selection."""

from typing import Sequence
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionObjective:
    """Maximize validation accuracy minus the selected-feature penalty."""

    def __init__(self, X, y, feature_weight: float = 0.05,
                 validation_size: float = 0.2, seed: int = 42):
        if not math.isfinite(feature_weight) or feature_weight < 0:
            raise ValueError("feature_weight must be finite and nonnegative.")
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a two-dimensional feature matrix.")
        self.dimensions = X.shape[1]
        self.feature_weight = feature_weight
        self.seed = seed
        self.X_train, self.X_validation, self.y_train, self.y_validation = (
            train_test_split(X, y, test_size=validation_size,
                             random_state=seed, stratify=y)
        )
        self.last_accuracy = None

    def __call__(self, position: Sequence[int]) -> float:
        mask = np.asarray(position)
        if mask.shape != (self.dimensions,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Each feature must have one binary selection bit.")
        selected = mask.astype(bool)
        self.last_accuracy = None
        if not selected.any():
            return -1.0 - self.feature_weight  # Worse than every valid subset.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=self.seed),
        )
        model.fit(self.X_train[:, selected], self.y_train)
        self.last_accuracy = float(model.score(
            self.X_validation[:, selected], self.y_validation))
        return (self.last_accuracy
                - self.feature_weight * selected.sum() / self.dimensions)
