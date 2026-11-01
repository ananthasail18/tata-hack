"""
Rule-Based Triage Agent for Connected Fleet Predictive Maintenance.
Pure deterministic Python decision rules without LLMs or external agent frameworks.
Combines RUL forecasts, anomaly severity, and critical sensor thresholds into actionable maintenance directives.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

class TriageAgent:
    """
    Deterministic rule-based maintenance triage agent.
    Maps physical degradation parameters to maintenance priority tiers.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            # Standard competition demo thresholds
            self.critical_rul = 15.0
            self.critical_anomaly = 0.70
            self.high_rul = 40.0
            self.high_anomaly = 0.55
            self.medium_rul = 80.0
            self.medium_anomaly = 0.40
        else:
            rules = config.get("rules", {})
            self.critical_rul = rules.get("critical", {}).get("max_rul", 15.0)
            self.critical_anomaly = rules.get("critical", {}).get("min_anomaly_score", 0.70)
            self.high_rul = rules.get("high", {}).get("max_rul", 40.0)
            self.high_anomaly = rules.get("high", {}).get("min_anomaly_score", 0.55)
            self.medium_rul = rules.get("medium", {}).get("max_rul", 80.0)
            self.medium_anomaly = rules.get("medium", {}).get("min_anomaly_score", 0.40)

    def evaluate(
        self,
        engine_id: int,
        predicted_rul: float,
        anomaly_score: float,
        anomaly_status: str,
        latest_sensors: Dict[str, float],
        model_version: str = "v1.0.0",
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate engine state and determine triage severity, reason, and recommended action.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Validation: check for corrupted or invalid inputs
        if predicted_rul is None or not isinstance(predicted_rul, (int, float)) or predicted_rul < 0:
            return {
                "engine_id": engine_id,
                "severity": "CRITICAL",
                "recommended_action": "Ground engine immediately for sensor validation and system diagnostic audit.",
                "reason": f"Invalid or negative RUL prediction ({predicted_rul}). Potential telemetry or model failure.",
                "work_order_required": True,
                "predicted_rul": predicted_rul,
                "anomaly_score": anomaly_score,
                "anomaly_status": anomaly_status,
                "model_version": model_version,
                "timestamp": timestamp
            }

        reasons = []
        # Check specific sensor thresholds (aerospace failure limits in FD001)
        s4_val = latest_sensors.get("sensor_4", None)   # T50 LPT Temp (baseline ~1402, limit ~1425)
        s11_val = latest_sensors.get("sensor_11", None) # Ps30 HPC Static Pressure (baseline ~47.3, limit ~48.0)
        s7_val = latest_sensors.get("sensor_7", None)   # P30 Total Pressure (baseline ~554, limit ~551)

        sensor_alerts = []
        if s4_val is not None and s4_val >= 1425.0:
            sensor_alerts.append(f"Elevated LPT exhaust gas temp T50 ({s4_val:.1f} deg R >= 1425.0)")
        if s11_val is not None and s11_val >= 48.0:
            sensor_alerts.append(f"High HPC static pressure Ps30 ({s11_val:.2f} psia >= 48.0)")
        if s7_val is not None and s7_val <= 551.5:
            sensor_alerts.append(f"Depressed HPC total discharge pressure P30 ({s7_val:.1f} psia <= 551.5)")

        # Decoupled Decision Logic combining RUL and Anomaly Score
        # Case A: Imminent Physical End-of-Life (True Degradation Failure)
        if predicted_rul <= self.critical_rul:
            severity = "CRITICAL"
            action = "Ground engine immediately for priority inspection and hot-section overhaul."
            reasons.append(f"Critical Remaining Useful Life: {predicted_rul:.1f} cycles remaining (<= {self.critical_rul}).")
            if anomaly_score >= self.high_anomaly:
                reasons.append(f"Corroborated by high anomaly score: {anomaly_score:.2f}.")
            work_order_required = True

        # Case B: Sensor / Instrumentation Anomaly (High Anomaly Score but High RUL)
        elif anomaly_score >= self.critical_anomaly and predicted_rul > self.medium_rul:
            severity = "CRITICAL"
            action = "Ground engine for urgent sensor harness and instrumentation audit. Statistically anomalous telemetry detected despite high RUL forecast."
            reasons.append(
                f"Statistical telemetry anomaly detected (anomaly score {anomaly_score:.2f} >= {self.critical_anomaly}) "
                f"while mechanical RUL forecast remains substantial ({predicted_rul:.1f} cycles > {self.medium_rul}). "
                f"Indicates probable instrumentation fault, transducer drift, or auxiliary subsystem irregularity rather than core HPC blade wear."
            )
            work_order_required = True

        # Case C: Extreme Anomaly with Moderate RUL
        elif anomaly_score >= self.critical_anomaly:
            severity = "CRITICAL"
            action = "Ground engine for comprehensive mechanical and instrumentation diagnostics."
            reasons.append(f"Extreme sensor anomaly score: {anomaly_score:.2f} with remaining life {predicted_rul:.1f} cycles.")
            work_order_required = True

        # Case D: High Degradation Window (15 < RUL <= 40)
        elif predicted_rul <= self.high_rul or anomaly_score >= self.high_anomaly:
            severity = "HIGH"
            action = "Prioritize maintenance within next scheduled flight rotation / 5 cycles."
            if predicted_rul <= self.high_rul:
                reasons.append(f"High degradation stage: {predicted_rul:.1f} cycles remaining (<= {self.high_rul}).")
            if anomaly_score >= self.high_anomaly:
                reasons.append(f"Elevated anomaly score: {anomaly_score:.2f} (>= {self.high_anomaly}).")
            work_order_required = True

        # Case E: Emerging Degradation Window (40 < RUL <= 80)
        elif predicted_rul <= self.medium_rul or anomaly_score >= self.medium_anomaly:
            severity = "MEDIUM"
            action = "Schedule workshop boroscope inspection and monitor trendline closely."
            if predicted_rul <= self.medium_rul:
                reasons.append(f"Emerging degradation detected: {predicted_rul:.1f} cycles remaining (<= {self.medium_rul}).")
            if anomaly_score >= self.medium_anomaly:
                reasons.append(f"Moderate telemetry irregularity: anomaly score {anomaly_score:.2f}.")
            work_order_required = True

        # Case F: Nominal Healthy Cruise
        else:
            severity = "LOW"
            action = "Routine continuous monitoring; engine operating within nominal design envelope."
            reasons.append(f"Engine operating nominally with {predicted_rul:.1f} estimated cycles remaining.")
            work_order_required = False

        if sensor_alerts:
            reasons.append("Telemetry warnings: " + "; ".join(sensor_alerts))

        return {
            "engine_id": int(engine_id),
            "severity": severity,
            "recommended_action": action,
            "reason": " ".join(reasons),
            "work_order_required": work_order_required,
            "predicted_rul": round(float(predicted_rul), 1),
            "anomaly_score": round(float(anomaly_score), 4),
            "anomaly_status": anomaly_status,
            "model_version": model_version,
            "timestamp": timestamp
        }
