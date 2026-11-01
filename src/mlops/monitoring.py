"""
Monitoring Module for Data Drift, Prediction Distribution, and Health Metrics.
Tracks statistical shifts in incoming telemetry versus the training baseline,
prediction distribution characteristics, and fleet anomaly frequencies.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class PipelineMonitor:
    """Monitors telemetry distributions, model forecasts, and anomaly occurrences."""
    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline_df = baseline_df

    def compute_data_drift(
        self,
        current_df: pd.DataFrame,
        sensor_cols: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Evaluate sensor distribution drift using the two-sample Kolmogorov-Smirnov (KS) test.
        Compares incoming telemetry against baseline training telemetry.
        """
        if self.baseline_df is None:
            return {"error": "No baseline dataframe provided for drift comparison."}

        if sensor_cols is None:
            sensor_cols = [
                "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
                "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                "sensor_17", "sensor_20", "sensor_21"
            ]

        drift_report = {}
        drifted_count = 0

        for col in sensor_cols:
            if col in self.baseline_df.columns and col in current_df.columns:
                stat, p_val = ks_2samp(self.baseline_df[col].dropna(), current_df[col].dropna())
                is_drift = bool(p_val < alpha)
                if is_drift:
                    drifted_count += 1
                drift_report[col] = {
                    "ks_statistic": round(float(stat), 4),
                    "p_value": round(float(p_val), 4),
                    "drift_detected": is_drift,
                    "baseline_mean": round(float(self.baseline_df[col].mean()), 2),
                    "current_mean": round(float(current_df[col].mean()), 2)
                }

        return {
            "total_sensors_checked": len(drift_report),
            "drifted_sensors_count": drifted_count,
            "drift_percentage": round((drifted_count / len(drift_report)) * 100, 1) if drift_report else 0.0,
            "sensor_details": drift_report
        }

    def compute_prediction_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Summarize output prediction distribution characteristics."""
        return {
            "count": int(len(predictions)),
            "mean": round(float(np.mean(predictions)), 2),
            "std": round(float(np.std(predictions)), 2),
            "min": round(float(np.min(predictions)), 2),
            "p25": round(float(np.percentile(predictions, 25)), 2),
            "median": round(float(np.median(predictions)), 2),
            "p75": round(float(np.percentile(predictions, 75)), 2),
            "max": round(float(np.max(predictions)), 2)
        }

    def compute_anomaly_frequency(self, anomaly_flags: np.ndarray) -> Dict[str, Any]:
        """Compute anomaly trigger rate across processed engine cycles."""
        total = len(anomaly_flags)
        anomalies = int(np.sum(anomaly_flags))
        rate = round((anomalies / total) * 100, 2) if total > 0 else 0.0
        return {
            "total_observations": total,
            "anomaly_count": anomalies,
            "anomaly_rate_pct": rate
        }
