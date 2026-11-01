"""
Maintenance Work Order Management Module for Connected Fleet.
Persists work orders in SQLite, manages work-order lifecycle, and prevents duplicate work orders.
"""

from datetime import datetime, timezone
import json
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

class WorkOrderManager:
    """
    SQLite-backed maintenance work order manager.
    Supports lifecycle tracking, status updates, review audit trail, and duplicate suppression.
    """
    def __init__(self, db_path: str = "data/work_orders.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Create work orders table and indexes if not exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS work_orders (
                    work_order_id TEXT PRIMARY KEY,
                    engine_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    predicted_rul REAL NOT NULL,
                    severity TEXT NOT NULL,
                    anomaly_status TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    important_sensors TEXT,
                    recommended_action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    reviewed_by TEXT,
                    reviewer_notes TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_engine_status 
                ON work_orders (engine_id, status)
            """)
            conn.commit()

    def get_active_work_order(self, engine_id: int) -> Optional[Dict[str, Any]]:
        """
        Check if an active (unresolved) work order already exists for the engine.
        Active statuses include 'OPEN', 'IN_REVIEW', and 'APPROVED'.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM work_orders 
                WHERE engine_id = ? AND status IN ('OPEN', 'IN_REVIEW', 'APPROVED')
                ORDER BY created_at DESC LIMIT 1
            """, (engine_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def create_work_order(
        self,
        triage_result: Dict[str, Any],
        sensors: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new maintenance work order from a TriageAgent evaluation.
        Suppresses creation if an active unresolved work order already exists for this engine.
        """
        if not triage_result.get("work_order_required", False):
            return None

        engine_id = triage_result["engine_id"]
        
        # Duplicate Prevention Check
        existing_active = self.get_active_work_order(engine_id)
        if existing_active:
            # If current severity is higher than existing, update existing reason/severity
            return existing_active

        wo_id = f"WO-ENG{engine_id:03d}-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        now_ts = datetime.now(timezone.utc).isoformat()
        
        sensors_json = json.dumps(sensors if sensors else {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO work_orders (
                    work_order_id, engine_id, created_at, predicted_rul,
                    severity, anomaly_status, anomaly_score, important_sensors,
                    recommended_action, reason, model_version, status,
                    reviewed_by, reviewer_notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', NULL, NULL, ?)
            """, (
                wo_id,
                engine_id,
                now_ts,
                triage_result["predicted_rul"],
                triage_result["severity"],
                triage_result["anomaly_status"],
                triage_result.get("anomaly_score", 0.0),
                sensors_json,
                triage_result["recommended_action"],
                triage_result["reason"],
                triage_result.get("model_version", "v1.0.0"),
                now_ts
            ))
            conn.commit()

        return self.get_work_order(wo_id)

    def get_work_order(self, work_order_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific work order by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM work_orders WHERE work_order_id = ?", (work_order_id,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                if res.get("important_sensors"):
                    try:
                        res["important_sensors"] = json.loads(res["important_sensors"])
                    except Exception:
                        pass
                return res
            return None

    def list_work_orders(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        engine_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query work orders with optional filtering."""
        query = "SELECT * FROM work_orders WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if engine_id is not None:
            query += " AND engine_id = ?"
            params.append(engine_id)

        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("important_sensors"):
                    try:
                        item["important_sensors"] = json.loads(item["important_sensors"])
                    except Exception:
                        pass
                results.append(item)
            return results

    def update_status(
        self,
        work_order_id: str,
        new_status: str,
        reviewed_by: Optional[str] = None,
        reviewer_notes: Optional[str] = None
    ) -> bool:
        """
        Update the status of a work order (e.g., 'IN_REVIEW', 'APPROVED', 'COMPLETED', 'CANCELLED').
        """
        allowed = ["OPEN", "IN_REVIEW", "APPROVED", "COMPLETED", "CANCELLED"]
        if new_status not in allowed:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {allowed}")

        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE work_orders 
                SET status = ?, 
                    reviewed_by = COALESCE(?, reviewed_by),
                    reviewer_notes = COALESCE(?, reviewer_notes),
                    updated_at = ?
                WHERE work_order_id = ?
            """, (new_status, reviewed_by, reviewer_notes, now_ts, work_order_id))
            conn.commit()
            return cursor.rowcount > 0

    def reset_database(self) -> None:
        """Clear all work orders (used for testing or resetting demonstration scenarios)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM work_orders")
            conn.commit()
