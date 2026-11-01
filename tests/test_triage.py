"""
Unit tests for the Rule-Based Triage Agent.
"""

import pytest
from src.agents.triage_agent import TriageAgent

def test_triage_critical_low_rul():
    agent = TriageAgent()
    res = agent.evaluate(
        engine_id=1,
        predicted_rul=8.0,
        anomaly_score=0.20,
        anomaly_status="NORMAL",
        latest_sensors={"sensor_4": 1428.0}
    )
    assert res["severity"] == "CRITICAL"
    assert res["work_order_required"] is True
    assert "Critical Remaining Useful Life" in res["reason"]

def test_triage_anomaly_override():
    """Even if RUL is relatively safe, high anomaly score must trigger escalation."""
    agent = TriageAgent()
    res = agent.evaluate(
        engine_id=2,
        predicted_rul=95.0,
        anomaly_score=0.75,
        anomaly_status="ANOMALY",
        latest_sensors={"sensor_4": 1405.0}
    )
    assert res["severity"] == "CRITICAL"
    assert res["work_order_required"] is True
    assert "anomaly score" in res["reason"].lower()

def test_triage_nominal_operation():
    agent = TriageAgent()
    res = agent.evaluate(
        engine_id=3,
        predicted_rul=110.0,
        anomaly_score=0.15,
        anomaly_status="NORMAL",
        latest_sensors={"sensor_4": 1402.0}
    )
    assert res["severity"] == "LOW"
    assert res["work_order_required"] is False

def test_triage_high_rul_sensor_anomaly():
    """High anomaly with high RUL must flag instrumentation audit rather than blade overhaul."""
    agent = TriageAgent()
    res = agent.evaluate(
        engine_id=5,
        predicted_rul=110.0,
        anomaly_score=0.82,
        anomaly_status="ANOMALY",
        latest_sensors={"sensor_4": 1404.0}
    )
    assert res["severity"] == "CRITICAL"
    assert res["work_order_required"] is True
    assert "sensor harness" in res["recommended_action"].lower()
    assert "instrumentation fault" in res["reason"].lower()

def test_triage_invalid_rul_handling():
    agent = TriageAgent()
    res = agent.evaluate(
        engine_id=4,
        predicted_rul=-5.0,
        anomaly_score=0.20,
        anomaly_status="NORMAL",
        latest_sensors={}
    )
    assert res["severity"] == "CRITICAL"
    assert "Invalid or negative RUL" in res["reason"]
