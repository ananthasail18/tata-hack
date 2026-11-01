"""
Unsupervised Anomaly Detection Module for C-MAPSS Turbofan Telemetry.
Uses an Isolation Forest to detect abnormal operating states without ground-truth anomaly labels.
Provides calibrated anomaly scores [0, 1] and threshold-based alert statuses.
"""

import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class EngineAnomalyDetector:
    """
    Unsupervised Anomaly Detector using Isolation Forest.
    Trained on early/healthy baseline cycles to establish the nominal operating envelope.
    """
    def __init__(self, contamination: float = 0.10, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.score_min: float = -0.5
        self.score_max: float = 0.5

    def fit(self, X_train: np.ndarray, healthy_mask: Optional[np.ndarray] = None) -> "EngineAnomalyDetector":
        """
        Fit Isolation Forest model. If healthy_mask is provided, fit predominantly
        on known early-cycle healthy telemetry to define nominal space.
        """
        if healthy_mask is not None and np.sum(healthy_mask) > 100:
            X_fit = X_train[healthy_mask]
        else:
            X_fit = X_train
            
        self.model.fit(X_fit)
        self.is_fitted = True
        
        # Calibration bounds from decision function on training set
        raw_scores = self.model.decision_function(X_train)
        self.score_min = float(np.percentile(raw_scores, 1))
        self.score_max = float(np.percentile(raw_scores, 99))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute calibrated anomaly scores in [0.0, 1.0].
        0.0 = completely nominal / baseline.
        1.0 = highly anomalous / severe outlier.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before scoring samples.")
            
        raw_decision = self.model.decision_function(X)
        # Invert: decision_function is positive for normal, negative for anomaly
        clipped = np.clip(raw_decision, self.score_min, self.score_max)
        normalized_score = 1.0 - (clipped - self.score_min) / (self.score_max - self.score_min + 1e-8)
        return np.round(normalized_score, 4)

    def predict(self, X: np.ndarray, threshold: float = 0.60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly flag and return calibrated score.
        
        Returns:
            (is_anomaly_bool_array, anomaly_scores_array)
        """
        scores = self.score_samples(X)
        is_anomaly = scores >= threshold
        return is_anomaly, scores

    def save(self, filepath: str = "models/anomaly_detector.joblib") -> None:
        """Serialize detector to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/anomaly_detector.joblib") -> "EngineAnomalyDetector":
        """Load serialized detector."""
        return joblib.load(filepath)
