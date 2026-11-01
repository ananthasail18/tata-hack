"""
Feature Engineering Module for C-MAPSS Telemetry.
Builds rolling statistics, first differences, trend estimates, and baseline deviations.
Guarantees NO future data leakage by computing past-only rolling operations per engine.
"""

import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeaturePipeline:
    """
    Stateful feature engineering and scaling pipeline for turbofan engine time series.
    Fitted strictly on training data to prevent fleet-level statistics leakage.
    """
    def __init__(
        self,
        active_sensors: Optional[List[str]] = None,
        rolling_windows: List[int] = [5, 10],
        baseline_cycles: int = 5
    ):
        if active_sensors is None:
            self.active_sensors = [
                "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
                "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                "sensor_17", "sensor_20", "sensor_21"
            ]
        else:
            self.active_sensors = active_sensors
            
        self.rolling_windows = rolling_windows
        self.baseline_cycles = baseline_cycles
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _extract_engine_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Extract past-only features for a single engine trajectory.
        Prevents any future information leakage across cycles.
        """
        group = group.sort_values("cycle").copy()
        feat_dict = {}
        
        # Keep identifier columns
        feat_dict["unit"] = group["unit"]
        feat_dict["cycle"] = group["cycle"]
        if "RUL" in group.columns:
            feat_dict["RUL"] = group["RUL"]
        if "RUL_capped" in group.columns:
            feat_dict["RUL_capped"] = group["RUL_capped"]

        # 1. Operational settings
        feat_dict["op_setting_1"] = group["op_setting_1"]
        feat_dict["op_setting_2"] = group["op_setting_2"]

        for s in self.active_sensors:
            vals = group[s]
            
            # Current raw measurement
            feat_dict[f"{s}_raw"] = vals

            # Rolling means (past-only)
            for w in self.rolling_windows:
                feat_dict[f"{s}_roll_mean_{w}"] = vals.rolling(window=w, min_periods=1).mean()

            # Rolling standard deviation (10 cycles)
            r_std = vals.rolling(window=10, min_periods=2).std().fillna(0.0)
            feat_dict[f"{s}_roll_std_10"] = r_std

            # First difference (instantaneous rate of change)
            feat_dict[f"{s}_diff_1"] = vals.diff().fillna(0.0)

            # Rolling short-term trend (difference between current value and past rolling mean)
            feat_dict[f"{s}_trend_5"] = vals - feat_dict[f"{s}_roll_mean_5"]

            # Engine-specific baseline deviation (deviation from initial healthy cycles)
            initial_baseline = vals.iloc[:self.baseline_cycles].mean()
            feat_dict[f"{s}_baseline_delta"] = vals - initial_baseline

        return pd.DataFrame(feat_dict, index=group.index)

    def transform_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features across all engines without scaling."""
        engine_dfs = []
        for _, grp in df.groupby("unit"):
            engine_dfs.append(self._extract_engine_features(grp))
        combined_df = pd.concat(engine_dfs, axis=0).reset_index(drop=True)
        return combined_df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Fit scaler on training data and return scaled feature matrix and target.
        """
        feat_df = self.transform_raw_dataframe(df)
        
        meta_cols = ["unit", "cycle", "RUL", "RUL_capped"]
        self.feature_names = [c for c in feat_df.columns if c not in meta_cols]
        
        X_raw = feat_df[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X_raw)
        
        self.is_fitted = True
        return X_scaled, feat_df["RUL"].values if "RUL" in feat_df.columns else None, feat_df

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """
        Transform new or validation dataframe using fitted training scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted on training data before calling transform.")
            
        feat_df = self.transform_raw_dataframe(df)
        X_raw = feat_df[self.feature_names].values
        X_scaled = self.scaler.transform(X_raw)
        
        y = feat_df["RUL"].values if "RUL" in feat_df.columns else None
        return X_scaled, y, feat_df

    def save(self, filepath: str = "models/feature_pipeline.joblib") -> None:
        """Serialize fitted pipeline artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/feature_pipeline.joblib") -> "FeaturePipeline":
        """Load serialized pipeline artifact."""
        return joblib.load(filepath)
