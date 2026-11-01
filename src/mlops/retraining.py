"""
Automated Retraining and Champion-Challenger Promotion Module.
Simulates new incoming engine degradation data, retrains candidate models,
evaluates against acceptance gates, and manages production model promotion/rollback.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.ingestion import load_cmapss_data
from src.data.preprocessing import calculate_training_rul, apply_rul_cap, split_engines_by_unit
from src.features.engineering import FeaturePipeline
from src.models.train import train_xgboost
from src.models.predict import RULPredictor
from src.models.evaluate import compute_regression_metrics
from src.mlops.registry import ModelRegistry
from src.mlops.tracking import MLflowTracker

def simulate_additional_engine_data(base_df: pd.DataFrame, n_additional: int = 5, random_seed: int = 99) -> pd.DataFrame:
    """
    Simulate newly arrived, fully run-to-failure engine trajectories
    by taking a sample of existing trajectories and adding slight variation.
    Used for demonstrating continuous retraining when new operational logs arrive.
    """
    np.random.seed(random_seed)
    available_units = base_df["unit"].unique()
    sample_units = np.random.choice(available_units, size=n_additional, replace=False)
    
    new_dfs = []
    max_existing_id = int(base_df["unit"].max())
    
    for i, orig_u in enumerate(sample_units):
        sub = base_df[base_df["unit"] == orig_u].copy()
        new_u = max_existing_id + i + 1
        sub["unit"] = new_u
        # Add slight sensor noise (±0.5%)
        sensor_cols = [c for c in sub.columns if c.startswith("sensor_")]
        for s in sensor_cols:
            noise = np.random.normal(0, 0.005 * sub[s].std() if sub[s].std() > 0 else 0.001, size=len(sub))
            sub[s] = sub[s] + noise
        new_dfs.append(sub)
        
    return pd.concat(new_dfs, axis=0).reset_index(drop=True)

def run_retraining_workflow(
    raw_data_dir: str = "data/raw",
    rul_cap: int = 125,
    acceptance_max_rmse: float = 25.0,
    force_promotion: bool = False
) -> Dict[str, Any]:
    """
    Executes an end-to-end retraining flow:
    1. Loads base data + simulates incoming engine telemetry.
    2. Builds features and splits train/validation.
    3. Trains challenger model.
    4. Evaluates challenger vs existing champion.
    5. Applies acceptance criteria and conditionally promotes.
    """
    registry = ModelRegistry()
    tracker = MLflowTracker()
    champion_meta = registry.get_champion()

    print("[Retraining] Loading base dataset and simulating incoming engine logs...")
    raw_df = load_cmapss_data(data_dir=raw_data_dir, subset="FD001", split="train")
    new_engines_df = simulate_additional_engine_data(raw_df, n_additional=5)
    augmented_df = pd.concat([raw_df, new_engines_df], axis=0).reset_index(drop=True)
    
    augmented_df = calculate_training_rul(augmented_df)
    augmented_df = apply_rul_cap(augmented_df, cap=rul_cap)

    # Train / Val split
    train_df, val_df, train_units, val_units = split_engines_by_unit(augmented_df, train_ratio=0.8, random_state=42)

    # Re-fit feature pipeline
    print("[Retraining] Fitting updated feature pipeline...")
    pipeline = FeaturePipeline()
    X_train, y_train, _ = pipeline.fit_transform(train_df)
    target_col = "RUL_capped" if rul_cap else "RUL"
    y_train = train_df[target_col].values

    X_val, _, val_feat_df = pipeline.transform(val_df)
    y_val_true = val_df["RUL"].values

    # Train candidate model
    print("[Retraining] Training candidate XGBoost model...")
    candidate_xgb = train_xgboost(X_train, y_train)
    candidate_predictor = RULPredictor(candidate_xgb, pipeline, model_type="xgboost", rul_cap=rul_cap)

    # Evaluate candidate
    y_val_pred = candidate_predictor.predict(val_df)
    candidate_metrics = compute_regression_metrics(y_val_true, y_val_pred)
    print(f"[Retraining] Candidate Metrics: {candidate_metrics}")

    # Evaluate current champion on same validation set if exists
    champion_metrics = None
    if champion_meta and os.path.exists(champion_meta["artifact_path"]):
        champ_predictor = RULPredictor.load(champion_meta["artifact_path"])
        y_champ_pred = champ_predictor.predict(val_df)
        champion_metrics = compute_regression_metrics(y_val_true, y_val_pred)
        print(f"[Retraining] Current Champion Metrics: {champion_metrics}")

    # Determine acceptance and promotion
    is_accepted = False
    rejection_reasons = []

    if candidate_metrics["rmse"] > acceptance_max_rmse:
        rejection_reasons.append(f"Candidate RMSE {candidate_metrics['rmse']} exceeds ceiling {acceptance_max_rmse}")

    if champion_metrics:
        # Candidate must not degrade performance significantly (>5% margin)
        if candidate_metrics["rmse"] > champion_metrics["rmse"] * 1.05:
            rejection_reasons.append(f"Candidate RMSE ({candidate_metrics['rmse']}) is worse than Champion ({champion_metrics['rmse']})")

    if len(rejection_reasons) == 0 or force_promotion:
        is_accepted = True

    # Version numbering
    prev_models = registry.list_models()
    new_version = f"v{len(prev_models) + 1}.0.0"

    # Save artifact
    candidate_path = f"models/xgboost_{new_version}.joblib"
    candidate_predictor.save(candidate_path)

    # Register
    reg_entry = registry.register_model(
        version=new_version,
        model_type="xgboost",
        artifact_path=candidate_path,
        metrics=candidate_metrics,
        hyperparameters={"rul_cap": rul_cap, "n_estimators": 120, "learning_rate": 0.05, "total_engines": augmented_df['unit'].nunique()},
        is_champion=is_accepted
    )

    # Log to MLflow
    tracker.log_run(
        run_name=f"Retraining_{new_version}",
        params={"model_version": new_version, "rul_cap": rul_cap, "total_engines": augmented_df['unit'].nunique()},
        metrics=candidate_metrics,
        tags={"promoted": str(is_accepted), "type": "retraining"}
    )

    result = {
        "status": "SUCCESS",
        "candidate_version": new_version,
        "is_promoted_to_champion": is_accepted,
        "rejection_reasons": rejection_reasons,
        "candidate_metrics": candidate_metrics,
        "previous_champion_metrics": champion_metrics,
        "total_training_engines": augmented_df["unit"].nunique()
    }
    return result
