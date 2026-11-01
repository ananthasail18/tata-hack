"""
Unit tests for Feature Pipeline and Data Leakage Prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_cmapss_data
from src.features.engineering import FeaturePipeline

def test_feature_pipeline_no_future_leakage():
    """
    Leakage Test: Verify that modifying future cycles of an engine
    does not alter any engineered feature values for earlier cycles!
    """
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    u1 = df[df["unit"] == 1].copy()
    
    pipeline = FeaturePipeline()
    feat_orig = pipeline.transform_raw_dataframe(u1)
    
    # Mutate sensor values only from cycle 100 onwards
    u1_mutated = u1.copy()
    sensor_cols = [c for c in u1_mutated.columns if c.startswith("sensor_")]
    u1_mutated.loc[u1_mutated["cycle"] >= 100, sensor_cols] += 500.0
    
    feat_mutated = pipeline.transform_raw_dataframe(u1_mutated)
    
    # Cycles 1 to 99 MUST BE 100% IDENTICAL
    early_orig = feat_orig[feat_orig["cycle"] < 100].drop(columns=["unit", "cycle", "RUL", "RUL_capped"], errors="ignore")
    early_mutated = feat_mutated[feat_mutated["cycle"] < 100].drop(columns=["unit", "cycle", "RUL", "RUL_capped"], errors="ignore")
    
    diff = np.abs(early_orig.values - early_mutated.values)
    max_diff = np.max(diff)
    assert max_diff < 1e-6, f"Data leakage detected! Future mutation leaked into earlier cycle features: max_diff = {max_diff}"

def test_feature_pipeline_fit_transform():
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    pipeline = FeaturePipeline()
    X_scaled, y, feat_df = pipeline.fit_transform(df.iloc[:500])
    
    assert X_scaled.shape[0] == 500
    assert len(pipeline.feature_names) > 50
    assert pipeline.is_fitted is True
    # Standard scaled features should have approximately zero mean
    assert np.all(np.abs(np.mean(X_scaled, axis=0)) < 0.1)
