"""
Integration test for the End-to-End Fleet Inference Pipeline.
"""

import os
import pytest
import pandas as pd

from src.data.ingestion import load_cmapss_data
from src.pipeline import run_fleet_inference
from src.maintenance.work_orders import WorkOrderManager

def test_fleet_inference_pipeline(tmp_path):
    test_raw = load_cmapss_data(data_dir="data/raw", subset="FD001", split="test")
    # Run on small subset of test engines (Units 1 to 5)
    sample_test = test_raw[test_raw["unit"].isin([1, 2, 3, 4, 5])].copy()
    
    # Use isolated test database in tmp_path
    test_db = os.path.join(tmp_path, "test_pipeline_wo.db")
    
    res = run_fleet_inference(
        sample_test,
        champion_path="models/champion_model.joblib",
        detector_path="models/anomaly_detector.joblib",
        db_path=test_db
    )
    
    assert res["total_engines"] == 5
    assert len(res["fleet_summary_df"]) == 5
    assert "severity_counts" in res
    assert "annotated_cycles_df" in res
    
    # Check that predictions are valid numbers
    assert not res["fleet_summary_df"]["predicted_rul"].isnull().any()
    assert (res["fleet_summary_df"]["predicted_rul"] >= 0).all()
    
    # Check that anomaly scores are calibrated in [0, 1]
    assert (res["fleet_summary_df"]["anomaly_score"] >= 0).all()
    assert (res["fleet_summary_df"]["anomaly_score"] <= 1.0).all()
    
    # Verify work order manager wrote to test_db
    wo_mgr = WorkOrderManager(db_path=test_db)
    wos = wo_mgr.list_work_orders()
    assert len(wos) >= 0
