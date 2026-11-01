"""
Unit tests for SQLite Work Order Management and Duplicate Prevention.
"""

import os
import pytest
from src.maintenance.work_orders import WorkOrderManager

@pytest.fixture
def temp_wo_manager(tmp_path):
    db_file = os.path.join(tmp_path, "test_work_orders.db")
    manager = WorkOrderManager(db_path=db_file)
    return manager

def test_work_order_creation_and_retrieval(temp_wo_manager):
    triage_res = {
        "engine_id": 42,
        "predicted_rul": 12.5,
        "severity": "CRITICAL",
        "anomaly_status": "ANOMALY",
        "anomaly_score": 0.72,
        "recommended_action": "Ground engine immediately.",
        "reason": "Critical RUL remaining.",
        "work_order_required": True,
        "model_version": "v1.0.0"
    }
    
    wo = temp_wo_manager.create_work_order(triage_res, sensors={"sensor_4": 1426.5})
    assert wo is not None
    assert wo["engine_id"] == 42
    assert wo["severity"] == "CRITICAL"
    assert wo["status"] == "OPEN"
    
    fetched = temp_wo_manager.get_work_order(wo["work_order_id"])
    assert fetched["work_order_id"] == wo["work_order_id"]
    assert fetched["important_sensors"]["sensor_4"] == 1426.5

def test_duplicate_work_order_prevention(temp_wo_manager):
    """Calling create_work_order multiple times for same engine must NOT create duplicate open orders!"""
    triage_res = {
        "engine_id": 99,
        "predicted_rul": 25.0,
        "severity": "HIGH",
        "anomaly_status": "NORMAL",
        "anomaly_score": 0.35,
        "recommended_action": "Schedule maintenance.",
        "reason": "High priority.",
        "work_order_required": True,
        "model_version": "v1.0.0"
    }
    
    wo1 = temp_wo_manager.create_work_order(triage_res)
    wo2 = temp_wo_manager.create_work_order(triage_res)
    
    # Must return the existing active work order without inserting a new one
    assert wo1["work_order_id"] == wo2["work_order_id"]
    
    all_wos = temp_wo_manager.list_work_orders(engine_id=99)
    assert len(all_wos) == 1

def test_work_order_status_update(temp_wo_manager):
    triage_res = {
        "engine_id": 15,
        "predicted_rul": 35.0,
        "severity": "HIGH",
        "anomaly_status": "NORMAL",
        "anomaly_score": 0.30,
        "recommended_action": "Inspection.",
        "reason": "Boroscope check.",
        "work_order_required": True,
        "model_version": "v1.0.0"
    }
    wo = temp_wo_manager.create_work_order(triage_res)
    
    # Update to IN_REVIEW
    temp_wo_manager.update_status(wo["work_order_id"], "IN_REVIEW", reviewed_by="Lead_Engineer", reviewer_notes="Reviewed sensor telemetry.")
    updated = temp_wo_manager.get_work_order(wo["work_order_id"])
    assert updated["status"] == "IN_REVIEW"
    assert updated["reviewed_by"] == "Lead_Engineer"
    
    # Update to APPROVED
    temp_wo_manager.update_status(wo["work_order_id"], "APPROVED")
    approved = temp_wo_manager.get_work_order(wo["work_order_id"])
    assert approved["status"] == "APPROVED"
