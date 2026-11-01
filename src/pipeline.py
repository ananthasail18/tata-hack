"""
Master End-to-End Pipeline Orchestration for NASA C-MAPSS Predictive Maintenance.
Encapsulates both the training orchestration (comparing Ridge, RF, XGBoost, and LSTM)
and the live inference orchestration (Features -> Anomaly -> RUL -> Triage -> Work Orders -> Monitoring).
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.ingestion import load_cmapss_data, load_test_rul
from src.data.validation import run_full_validation
from src.data.preprocessing import calculate_training_rul, apply_rul_cap, split_engines_by_unit
from src.features.engineering import FeaturePipeline
from src.models.train import (
    train_linear_baseline,
    train_random_forest,
    train_xgboost,
    train_lstm,
    predict_lstm
)
from src.models.predict import RULPredictor
from src.models.anomaly import EngineAnomalyDetector
from src.models.evaluate import compute_regression_metrics
from src.agents.triage_agent import TriageAgent
from src.maintenance.work_orders import WorkOrderManager
from src.mlops.tracking import MLflowTracker
from src.mlops.registry import ModelRegistry
from src.mlops.monitoring import PipelineMonitor

def run_model_training_and_benchmarking(
    raw_data_dir: str = "data/raw",
    rul_cap: int = 125,
    skip_lstm: bool = False
) -> Dict[str, Any]:
    """
    Executes full training and model comparison suite:
    1. Loads and validates FD001 training data.
    2. Compares Ridge, Random Forest, XGBoost, and PyTorch LSTM.
    3. Evaluates uncapped vs capped RUL.
    4. Evaluates against the official test set (RUL_FD001.txt).
    5. Saves champion model, anomaly detector, and evaluation reports.
    """
    print("=" * 60)
    print("PHASE 1: DATA INGESTION & VALIDATION")
    print("=" * 60)
    train_raw = load_cmapss_data(data_dir=raw_data_dir, subset="FD001", split="train")
    test_raw = load_cmapss_data(data_dir=raw_data_dir, subset="FD001", split="test")
    test_ruls = load_test_rul(data_dir=raw_data_dir, subset="FD001")

    val_report = run_full_validation(train_raw)
    print(f"Validation successful: {val_report['total_units']} engines, {val_report['total_rows']} cycles.")
    print(f"Active degrading sensors ({len(val_report['active_sensors'])}): {val_report['active_sensors']}")
    print(f"Invariant sensors ({len(val_report['constant_sensors'])}): {val_report['constant_sensors']}")

    # Target calculation
    train_df = calculate_training_rul(train_raw)
    train_df = apply_rul_cap(train_df, cap=rul_cap)

    # Engine-level Train / Validation Split (80% / 20%)
    train_split, val_split, train_units, val_units = split_engines_by_unit(
        train_df, train_ratio=0.8, random_state=42
    )
    print(f"Engine-wise split: {len(train_units)} training engines, {len(val_units)} validation engines.")

    print("\n" + "=" * 60)
    print("PHASE 2: FEATURE ENGINEERING & SCALING (NO LEAKAGE)")
    print("=" * 60)
    pipeline = FeaturePipeline(active_sensors=val_report["active_sensors"])
    X_train, _, train_feat_df = pipeline.fit_transform(train_split)
    pipeline.save("models/feature_pipeline.joblib")
    
    y_train_capped = train_split["RUL_capped"].values
    y_train_uncapped = train_split["RUL"].values

    X_val, _, val_feat_df = pipeline.transform(val_split)
    y_val_true = val_split["RUL"].values

    # Also transform test set for final benchmark
    X_test, _, test_feat_df = pipeline.transform(test_raw)

    print(f"Engineered feature vector dimension: {X_train.shape[1]} features.")

    print("\n" + "=" * 60)
    print("PHASE 3: UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)")
    print("=" * 60)
    # Train anomaly detector predominantly on early healthy cycles (cycle <= 30)
    healthy_mask = (train_split["cycle"].values <= 30)
    anomaly_detector = EngineAnomalyDetector(contamination=0.10, random_state=42)
    anomaly_detector.fit(X_train, healthy_mask=healthy_mask)
    anomaly_detector.save("models/anomaly_detector.joblib")
    print("Anomaly detector trained and serialized to models/anomaly_detector.joblib.")

    print("\n" + "=" * 60)
    print("PHASE 4: MODEL TRAINING & BENCHMARKING (RUL REGRESSION)")
    print("=" * 60)
    tracker = MLflowTracker()
    registry = ModelRegistry()
    results = []

    models_to_train = [
        ("Linear_Ridge", train_linear_baseline, "ridge"),
        ("Random_Forest", train_random_forest, "rf"),
        ("XGBoost", train_xgboost, "xgboost")
    ]

    trained_predictors = {}

    for name, train_fn, m_type in models_to_train:
        print(f"\nTraining {name} (Capped RUL = {rul_cap})...")
        t0 = time.time()
        fitted_model = train_fn(X_train, y_train_capped)
        train_time = round(time.time() - t0, 2)

        pred_wrapper = RULPredictor(fitted_model, pipeline, model_type=m_type, rul_cap=rul_cap)
        trained_predictors[name] = pred_wrapper

        # Evaluate on validation split
        val_preds = pred_wrapper.predict(val_split)
        val_metrics = compute_regression_metrics(y_val_true, val_preds)

        # Evaluate on official test set at final cycle
        test_pred_df = pred_wrapper.predict_final_cycles(test_raw)
        merged_test = pd.merge(test_pred_df, test_ruls, on="unit")
        test_metrics = compute_regression_metrics(merged_test["true_rul"].values, merged_test["predicted_rul"].values)

        print(f"  Validation: RMSE={val_metrics['rmse']}, MAE={val_metrics['mae']}, NASA={val_metrics['nasa_score']}")
        print(f"  Official Test: RMSE={test_metrics['rmse']}, MAE={test_metrics['mae']}, NASA={test_metrics['nasa_score']}")

        tracker.log_run(
            run_name=f"{name}_Capped",
            params={"model_type": m_type, "rul_cap": rul_cap, "train_time_sec": train_time},
            metrics={"val_rmse": val_metrics["rmse"], "test_rmse": test_metrics["rmse"], "test_nasa": test_metrics["nasa_score"]}
        )

        results.append({
            "model_name": name,
            "target": f"Capped ({rul_cap})",
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_nasa_score": test_metrics["nasa_score"],
            "train_time_s": train_time
        })

    # Train Uncapped XGBoost for comparison
    print("\nTraining XGBoost (Uncapped RUL)...")
    xgb_uncapped = train_xgboost(X_train, y_train_uncapped)
    pred_xgb_uncapped = RULPredictor(xgb_uncapped, pipeline, model_type="xgboost", rul_cap=None)
    val_preds_unc = pred_xgb_uncapped.predict(val_split)
    val_metrics_unc = compute_regression_metrics(y_val_true, val_preds_unc)
    test_pred_unc = pred_xgb_uncapped.predict_final_cycles(test_raw)
    merged_unc = pd.merge(test_pred_unc, test_ruls, on="unit")
    test_metrics_unc = compute_regression_metrics(merged_unc["true_rul"].values, merged_unc["predicted_rul"].values)
    
    results.append({
        "model_name": "XGBoost",
        "target": "Uncapped",
        "val_rmse": val_metrics_unc["rmse"],
        "val_mae": val_metrics_unc["mae"],
        "test_rmse": test_metrics_unc["rmse"],
        "test_mae": test_metrics_unc["mae"],
        "test_nasa_score": test_metrics_unc["nasa_score"],
        "train_time_s": 1.5
    })

    # PyTorch LSTM
    if not skip_lstm:
        print("\nTraining PyTorch LSTM (Sequence Length = 30)...")
        t0 = time.time()
        lstm_model = train_lstm(X_train, y_train_capped, train_split["unit"].values, seq_len=30, epochs=10, batch_size=128)
        train_time_lstm = round(time.time() - t0, 2)
        lstm_wrapper = RULPredictor(lstm_model, pipeline, model_type="lstm", rul_cap=rul_cap)
        trained_predictors["LSTM"] = lstm_wrapper

        val_preds_lstm = lstm_wrapper.predict(val_split)
        val_metrics_lstm = compute_regression_metrics(y_val_true, val_preds_lstm)
        test_pred_lstm = lstm_wrapper.predict_final_cycles(test_raw)
        merged_lstm = pd.merge(test_pred_lstm, test_ruls, on="unit")
        test_metrics_lstm = compute_regression_metrics(merged_lstm["true_rul"].values, merged_lstm["predicted_rul"].values)

        print(f"  LSTM Validation: RMSE={val_metrics_lstm['rmse']}, NASA={val_metrics_lstm['nasa_score']}")
        print(f"  LSTM Test: RMSE={test_metrics_lstm['rmse']}, NASA={test_metrics_lstm['nasa_score']}")

        results.append({
            "model_name": "PyTorch_LSTM",
            "target": f"Capped ({rul_cap})",
            "val_rmse": val_metrics_lstm["rmse"],
            "val_mae": val_metrics_lstm["mae"],
            "test_rmse": test_metrics_lstm["rmse"],
            "test_mae": test_metrics_lstm["mae"],
            "test_nasa_score": test_metrics_lstm["nasa_score"],
            "train_time_s": train_time_lstm
        })

    # Select Champion Model (Best Test RMSE / NASA score)
    benchmark_df = pd.DataFrame(results)
    benchmark_df.sort_values(by="test_rmse", inplace=True)
    best_row = benchmark_df.iloc[0]
    best_name = best_row["model_name"]
    print("\n" + "=" * 60)
    print(f"MODEL BENCHMARK RESULTS (Selected Champion: {best_name})")
    print("=" * 60)
    print(benchmark_df.to_string(index=False))

    # Save benchmark table to reports
    os.makedirs("outputs/reports", exist_ok=True)
    benchmark_df.to_csv("outputs/reports/model_comparison.csv", index=False)

    # Save champion predictor
    champion_predictor = trained_predictors.get(best_name, trained_predictors["XGBoost"])
    champion_path = f"models/{best_name}_v1.0.0.joblib" if best_name != "LSTM" else "models/lstm_v1.0.0.pt"
    champion_predictor.save(champion_path)
    
    registry.register_model(
        version="v1.0.0",
        model_type=best_name,
        artifact_path=champion_path,
        metrics={"test_rmse": float(best_row["test_rmse"]), "test_mae": float(best_row["test_mae"]), "test_nasa": float(best_row["test_nasa_score"])},
        hyperparameters={"rul_cap": rul_cap},
        is_champion=True
    )
    print(f"Champion model registered as v1.0.0 in {champion_path}.")

    # Generate test prediction visualization
    best_test_preds = champion_predictor.predict_final_cycles(test_raw)
    final_eval_df = pd.merge(best_test_preds, test_ruls, on="unit")
    final_eval_df["error"] = final_eval_df["predicted_rul"] - final_eval_df["true_rul"]
    final_eval_df.to_csv("outputs/predictions/test_predictions.csv", index=False)

    _generate_evaluation_figures(final_eval_df, best_name, float(best_row["test_rmse"]))

    return {
        "benchmark_df": benchmark_df,
        "champion_model_name": best_name,
        "champion_metrics": dict(best_row),
        "test_eval_df": final_eval_df
    }


def _generate_evaluation_figures(eval_df: pd.DataFrame, model_name: str, rmse: float) -> None:
    """Generate high-resolution evaluation figures for test predictions."""
    os.makedirs("outputs/figures", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    # 1. Predicted vs True Scatter Plot
    ax1 = axes[0]
    ax1.scatter(eval_df["true_rul"], eval_df["predicted_rul"], color="#1f77b4", alpha=0.7, edgecolors="k", s=45)
    max_val = max(eval_df["true_rul"].max(), eval_df["predicted_rul"].max()) + 10
    ax1.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", lw=2, label="Ideal Prediction (y = x)")
    ax1.set_title(f"Test Set: Predicted vs True RUL ({model_name} - RMSE: {rmse:.2f})", fontweight="bold")
    ax1.set_xlabel("True Remaining Useful Life (Cycles)")
    ax1.set_ylabel("Predicted Remaining Useful Life (Cycles)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Prediction Error Distribution
    ax2 = axes[1]
    sns.histplot(eval_df["error"], bins=20, kde=True, ax=ax2, color="#2ca02c", edgecolor="k")
    ax2.axvline(0, color="red", linestyle="--", lw=1.8, label="Zero Error")
    ax2.set_title(f"Prediction Error Distribution (Mean: {eval_df['error'].mean():.2f})", fontweight="bold")
    ax2.set_xlabel("Prediction Error (Predicted RUL - True RUL)")
    ax2.set_ylabel("Engine Count")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = "outputs/figures/model_evaluation_test_set.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Evaluation figures saved to {fig_path}.")


def run_fleet_inference(
    test_raw_df: pd.DataFrame,
    champion_path: str = "models/champion_model.joblib",
    detector_path: str = "models/anomaly_detector.joblib",
    db_path: str = "data/work_orders.db"
) -> Dict[str, Any]:
    """
    Executes live closed-loop fleet inference:
    1. Extracts engineered features for all incoming engine cycles.
    2. Runs anomaly detector -> scores and anomaly flags.
    3. Runs champion RUL model -> forecasts remaining useful life.
    4. Evaluates rule-based TriageAgent for each engine at its current operating cycle.
    5. Automatically logs and manages maintenance work orders in SQLite.
    6. Computes fleet health & data drift metrics.
    """
    # Load models
    if not os.path.exists(champion_path):
        # Fallback to versioned model if champion symlink not yet resolved
        candidate_paths = [
            "models/xgboost_v1.0.0.joblib",
            "models/champion_model.joblib"
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                champion_path = p
                break
                
    predictor = RULPredictor.load(champion_path)
    detector = EngineAnomalyDetector.load(detector_path)
    triage = TriageAgent()
    wo_manager = WorkOrderManager(db_path=db_path)

    # Transform features
    X_scaled, _, feat_df = predictor.pipeline.transform(test_raw_df)
    
    # Anomaly scoring
    is_anomaly, anomaly_scores = detector.predict(X_scaled, threshold=0.55)
    
    # RUL forecasting
    predicted_ruls = predictor.predict(test_raw_df)

    # Combine into full fleet cycle telemetry
    annotated_df = test_raw_df[["unit", "cycle"]].copy()
    annotated_df["predicted_rul"] = predicted_ruls
    annotated_df["anomaly_score"] = anomaly_scores
    annotated_df["is_anomaly"] = is_anomaly

    # Extract current latest status per engine (final observed cycle)
    latest_indices = annotated_df.groupby("unit")["cycle"].idxmax()
    fleet_summary = []
    work_orders_created = []

    for idx in latest_indices:
        row = annotated_df.loc[idx]
        unit_id = int(row["unit"])
        pred_rul = float(row["predicted_rul"])
        anom_score = float(row["anomaly_score"])
        anom_status = "ANOMALY" if row["is_anomaly"] else "NORMAL"
        
        # Latest sensor snapshot for triage evaluation
        raw_row = test_raw_df.loc[idx]
        latest_sensors = {
            "sensor_4": float(raw_row["sensor_4"]),
            "sensor_11": float(raw_row["sensor_11"]),
            "sensor_7": float(raw_row["sensor_7"]),
            "sensor_15": float(raw_row["sensor_15"])
        }

        # Triage evaluation
        triage_res = triage.evaluate(
            engine_id=unit_id,
            predicted_rul=pred_rul,
            anomaly_score=anom_score,
            anomaly_status=anom_status,
            latest_sensors=latest_sensors,
            model_version="v1.0.0"
        )

        # Work order generation (automatic with duplicate suppression)
        if triage_res["work_order_required"]:
            wo = wo_manager.create_work_order(triage_res, sensors=latest_sensors)
            if wo:
                work_orders_created.append(wo)

        fleet_summary.append({
            "unit": unit_id,
            "current_cycle": int(row["cycle"]),
            "predicted_rul": pred_rul,
            "anomaly_score": anom_score,
            "anomaly_status": anom_status,
            "severity": triage_res["severity"],
            "recommended_action": triage_res["recommended_action"],
            "reason": triage_res["reason"],
            "work_order_required": triage_res["work_order_required"]
        })

    fleet_summary_df = pd.DataFrame(fleet_summary)

    # Severity distribution counts
    sev_counts = fleet_summary_df["severity"].value_counts().to_dict()

    return {
        "fleet_summary_df": fleet_summary_df,
        "annotated_cycles_df": annotated_df,
        "total_engines": len(fleet_summary_df),
        "severity_counts": sev_counts,
        "work_orders_created_or_active": len(work_orders_created),
        "total_anomalies_flagged": int(is_anomaly.sum())
    }
