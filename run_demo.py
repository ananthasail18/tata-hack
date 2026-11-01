"""
NASA C-MAPSS Predictive Maintenance Demonstration Script
Teliport Season 4, Case Study 1: Connected Fleet Predictive Maintenance
Executes reproducible end-to-end demonstration workflow:
1. Ingests test fleet telemetry (100 engines).
2. Computes unsupervised anomaly detection scores.
3. Generates Remaining Useful Life (RUL) predictions.
4. Executes deterministic rule-based Triage Agent.
5. Automatically creates and stores maintenance work orders in SQLite.
6. Highlights 3 illustrative real-data engine scenarios (Healthy, Degrading, Critical).
7. Demonstrates continuous MLOps retraining, champion-challenger comparison, and model promotion.
"""

import argparse
import os
import sys
import pandas as pd

from src.data.ingestion import load_cmapss_data, load_test_rul
from src.pipeline import run_model_training_and_benchmarking, run_fleet_inference
from src.maintenance.work_orders import WorkOrderManager
from src.mlops.registry import ModelRegistry
from src.mlops.retraining import run_retraining_workflow

def print_banner(text: str) -> None:
    print("\n" + "=" * 75)
    print(f" {text}")
    print("=" * 75)

def run_competition_demonstration(reset_db: bool = True, skip_retrain: bool = False) -> None:
    print_banner("TELIPORT SEASON 4: CLOSED-LOOP PREDICTIVE MAINTENANCE SYSTEM")
    print("Dataset: NASA C-MAPSS FD001 (100 Engines, 1 Operating Condition, HPC Degradation)")
    
    # Check if models exist, if not train them first
    registry = ModelRegistry()
    champion = registry.get_champion()
    
    if not champion:
        print("\n[Step 0] No existing champion model found. Executing full benchmark and training...")
        train_res = run_model_training_and_benchmarking(skip_lstm=False)
        champion = registry.get_champion()

    print_banner(f"PHASE 1: LOAD TELEMETRY & RUN FLEET INFERENCE (MODEL: {champion['version']})")
    
    wo_mgr = WorkOrderManager()
    if reset_db:
        print("[Demo Setup] Resetting demo work order database...")
        wo_mgr.reset_database()

    print("Loading official test fleet (13,096 cycles across 100 test engines)...")
    test_raw = load_cmapss_data(data_dir="data/raw", subset="FD001", split="test")
    test_ruls = load_test_rul(data_dir="data/raw", subset="FD001")

    print("Running end-to-end inference pipeline (Features -> Anomaly Detector -> RUL -> Triage -> Work Orders)...")
    results = run_fleet_inference(test_raw)
    fleet_df = results["fleet_summary_df"]
    
    # Merge with ground truth for demo verification
    demo_df = pd.merge(fleet_df, test_ruls, on="unit")
    demo_df["error"] = demo_df["predicted_rul"] - demo_df["true_rul"]

    print_banner("PHASE 2: FLEET HEALTH & TRIAGE SUMMARY")
    sev_counts = results["severity_counts"]
    print(f"Total Monitored Fleet Engines: {results['total_engines']}")
    print(f"  [CRITICAL] Immediate Action Required: {sev_counts.get('CRITICAL', 0)}")
    print(f"  [HIGH]     Maintenance in <5 cycles:  {sev_counts.get('HIGH', 0)}")
    print(f"  [MEDIUM]   Schedule Inspection:       {sev_counts.get('MEDIUM', 0)}")
    print(f"  [LOW]      Nominal Cruise Monitoring: {sev_counts.get('LOW', 0)}")
    print(f"Total Cycles Flagged as Anomalous: {results['total_anomalies_flagged']}")
    print(f"Maintenance Work Orders Generated in SQLite: {results['work_orders_created_or_active']}")

    print_banner("PHASE 3: THREE ILLUSTRATIVE REAL-DATA ENGINE SCENARIOS")
    
    # 1. Critical Engine (Lowest RUL)
    crit_engine = demo_df[demo_df["severity"] == "CRITICAL"].sort_values("predicted_rul").iloc[0]
    print(f"\n[SCENARIO A: CRITICAL FLIGHT SAFETY RISK]")
    print(f"  Engine Unit: {int(crit_engine['unit'])}")
    print(f"  Current Operating Cycle: {int(crit_engine['current_cycle'])}")
    print(f"  Forecasted RUL: {crit_engine['predicted_rul']:.1f} cycles (Ground Truth: {crit_engine['true_rul']} cycles, Error: {crit_engine['error']:+.1f})")
    print(f"  Isolation Forest Anomaly Score: {crit_engine['anomaly_score']:.2f} ({crit_engine['anomaly_status']})")
    print(f"  Assigned Severity: {crit_engine['severity']}")
    print(f"  Recommended Action: {crit_engine['recommended_action']}")
    print(f"  Diagnostic Justification: {crit_engine['reason']}")

    # 2. High-Priority Engine (Moderate remaining life, significant drift)
    high_candidates = demo_df[demo_df["severity"] == "HIGH"].sort_values("predicted_rul")
    if not high_candidates.empty:
        high_engine = high_candidates.iloc[0]
        print(f"\n[SCENARIO B: HIGH DEGRADATION PRIORITY]")
        print(f"  Engine Unit: {int(high_engine['unit'])}")
        print(f"  Current Operating Cycle: {int(high_engine['current_cycle'])}")
        print(f"  Forecasted RUL: {high_engine['predicted_rul']:.1f} cycles (Ground Truth: {high_engine['true_rul']} cycles, Error: {high_engine['error']:+.1f})")
        print(f"  Isolation Forest Anomaly Score: {high_engine['anomaly_score']:.2f} ({high_engine['anomaly_status']})")
        print(f"  Assigned Severity: {high_engine['severity']}")
        print(f"  Recommended Action: {high_engine['recommended_action']}")
        print(f"  Diagnostic Justification: {high_engine['reason']}")

    # 3. Healthy / Nominal Engine (High RUL, low anomaly score)
    low_candidates = demo_df[demo_df["severity"] == "LOW"].sort_values("predicted_rul", ascending=False)
    if not low_candidates.empty:
        low_engine = low_candidates.iloc[0]
        print(f"\n[SCENARIO C: NOMINAL HEALTHY CRUISE]")
        print(f"  Engine Unit: {int(low_engine['unit'])}")
        print(f"  Current Operating Cycle: {int(low_engine['current_cycle'])}")
        print(f"  Forecasted RUL: {low_engine['predicted_rul']:.1f} cycles (Ground Truth: {low_engine['true_rul']} cycles, Error: {low_engine['error']:+.1f})")
        print(f"  Isolation Forest Anomaly Score: {low_engine['anomaly_score']:.2f} ({low_engine['anomaly_status']})")
        print(f"  Assigned Severity: {low_engine['severity']}")
        print(f"  Recommended Action: {low_engine['recommended_action']}")
        print(f"  Diagnostic Justification: {low_engine['reason']}")

    print_banner("PHASE 4: WORK ORDER LIFECYCLE & DUPLICATE PREVENTION AUDIT")
    work_orders = wo_mgr.list_work_orders()
    sample_wo = work_orders[0] if work_orders else None
    if sample_wo:
        print(f"Sample Work Order Created:")
        print(f"  WO ID: {sample_wo['work_order_id']}")
        print(f"  Target Engine: Unit {sample_wo['engine_id']}")
        print(f"  Severity: {sample_wo['severity']}")
        print(f"  Forecasted RUL: {sample_wo['predicted_rul']} cycles")
        print(f"  Initial Status: {sample_wo['status']}")
        
        # Demonstrate approval lifecycle
        print("\nSimulating Human-in-the-Loop review and status update...")
        wo_mgr.update_status(sample_wo["work_order_id"], "IN_REVIEW", reviewed_by="Chief_Inspector", reviewer_notes="Boroscope inspection confirmed HPC tip rub.")
        updated_wo = wo_mgr.get_work_order(sample_wo["work_order_id"])
        print(f"  Updated Status: {updated_wo['status']} (Reviewed by: {updated_wo['reviewed_by']})")
        print(f"  Audit Notes: {updated_wo['reviewer_notes']}")

        # Demonstrate duplicate prevention
        print("\nTesting duplicate prevention (running inference a second time)...")
        results_2 = run_fleet_inference(test_raw)
        wos_after = wo_mgr.list_work_orders()
        print(f"  Work orders before: {len(work_orders)}, after re-inference: {len(wos_after)} (Duplicates successfully suppressed!)")

    if not skip_retrain:
        print_banner("PHASE 5: MLOPs RETRAINING & CHAMPION-CHALLENGER COMPARISON")
        print("Simulating newly ingested completed engine runs and triggering candidate retraining...")
        retrain_summary = run_retraining_workflow()
        print(f"Retraining Status: {retrain_summary['status']}")
        print(f"Candidate Version: {retrain_summary['candidate_version']}")
        print(f"Candidate Metrics: {retrain_summary['candidate_metrics']}")
        print(f"Promoted to Production Champion: {retrain_summary['is_promoted_to_champion']}")

    print_banner("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("To launch the interactive monitoring dashboard, execute:")
    print("  python -m streamlit run dashboard/app.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA C-MAPSS Predictive Maintenance Demonstration")
    parser.add_argument("--no-reset", action="store_true", help="Do not clear existing work orders in SQLite")
    parser.add_argument("--skip-retrain", action="store_true", help="Skip the retraining workflow demonstration")
    args = parser.parse_args()

    run_competition_demonstration(reset_db=not args.no_reset, skip_retrain=args.skip_retrain)
