"""
Comprehensive Test Set Prediction and Ground-Truth Validation Script
Evaluates model predictions on official C-MAPSS test files against RUL ground truth.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ingestion import load_cmapss_data, load_test_rul
from src.models.predict import RULPredictor
from src.models.evaluate import compute_regression_metrics, nasa_scoring_function

def evaluate_subset_test_predictions(
    subset: str = "FD001",
    model_path: str = "models/champion_model.joblib",
    output_dir: str = "outputs/predictions"
) -> pd.DataFrame:
    print("=" * 80)
    print(f"OFFICIAL TEST SET VALIDATION: NASA C-MAPSS {subset}")
    print(f"Test Telemetry File: test_{subset}.txt")
    print(f"Ground Truth File:   RUL_{subset}.txt")
    print(f"Predictor Model:     {model_path}")
    print("=" * 80)

    # 1. Load test data and ground truth
    test_raw = load_cmapss_data(data_dir=".", subset=subset, split="test")
    test_rul = load_test_rul(data_dir=".", subset=subset)

    # 2. Load trained predictor
    predictor = RULPredictor.load(model_path)

    # 3. Generate predictions for the final observed operational cycle of each test engine
    preds_df = predictor.predict_final_cycles(test_raw)

    # 4. Merge predictions with official ground truth
    eval_df = pd.merge(preds_df, test_rul, on="unit")
    eval_df["error"] = eval_df["predicted_rul"] - eval_df["true_rul"]
    eval_df["abs_error"] = eval_df["error"].abs()
    eval_df["pct_error"] = (eval_df["abs_error"] / eval_df["true_rul"] * 100).round(1)

    # 5. Compute NASA asymmetric score component per engine
    def calc_nasa_component(e):
        return np.exp(-e / 13.0) - 1.0 if e < 0 else np.exp(e / 10.0) - 1.0
    eval_df["nasa_penalty"] = eval_df["error"].apply(calc_nasa_component).round(2)

    # 6. Assign diagnostic status category
    def categorize_prediction(e):
        if abs(e) <= 5.0:
            return "EXCELLENT (<= 5 cycles)"
        elif abs(e) <= 10.0:
            return "GOOD (<= 10 cycles)"
        elif abs(e) <= 15.0:
            return "ACCEPTABLE (<= 15 cycles)"
        elif e < -15.0:
            return "EARLY / CONSERVATIVE (Safe)"
        else:
            return "LATE / OVER-PREDICTION"
    eval_df["prediction_category"] = eval_df["error"].apply(categorize_prediction)

    # 7. Aggregate performance metrics
    y_true = eval_df["true_rul"].values
    y_pred = eval_df["predicted_rul"].values
    errors = eval_df["error"].values

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    nasa_score = nasa_scoring_function(y_true, y_pred)
    bias = np.mean(errors)
    median_ae = np.median(eval_df["abs_error"])

    pct_5 = np.mean(eval_df["abs_error"] <= 5) * 100
    pct_10 = np.mean(eval_df["abs_error"] <= 10) * 100
    pct_15 = np.mean(eval_df["abs_error"] <= 15) * 100
    pct_20 = np.mean(eval_df["abs_error"] <= 20) * 100

    print("\n" + "=" * 80)
    print("AGGREGATE BENCHMARK METRICS ON OFFICIAL TEST SET:")
    print("=" * 80)
    print(f"Total Test Engines Evaluated: {len(eval_df)}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} cycles")
    print(f"Mean Absolute Error (MAE):     {mae:.3f} cycles")
    print(f"NASA Asymmetric Score:          {nasa_score:.2f}")
    print(f"Mean Forecast Bias:             {bias:+.3f} cycles ({'Conservative / Early' if bias < 0 else 'Late'})")
    print(f"Median Absolute Error:          {median_ae:.3f} cycles")
    print("-" * 80)
    print("ACCURACY DISTRIBUTION:")
    print(f"  Within +/-  5 cycles: {pct_5:5.1f}% ({np.sum(eval_df['abs_error'] <= 5)} / {len(eval_df)} engines)")
    print(f"  Within +/- 10 cycles: {pct_10:5.1f}% ({np.sum(eval_df['abs_error'] <= 10)} / {len(eval_df)} engines)")
    print(f"  Within +/- 15 cycles: {pct_15:5.1f}% ({np.sum(eval_df['abs_error'] <= 15)} / {len(eval_df)} engines)")
    print(f"  Within +/- 20 cycles: {pct_20:5.1f}% ({np.sum(eval_df['abs_error'] <= 20)} / {len(eval_df)} engines)")

    # Breakdown by RUL Regime
    print("\n" + "=" * 80)
    print("PERFORMANCE BREAKDOWN BY RUL OPERATIONAL REGIME:")
    print("=" * 80)
    regimes = [
        ("Critical (RUL <= 15)", eval_df[eval_df["true_rul"] <= 15]),
        ("High Degradation (15 < RUL <= 40)", eval_df[(eval_df["true_rul"] > 15) & (eval_df["true_rul"] <= 40)]),
        ("Medium Degradation (40 < RUL <= 80)", eval_df[(eval_df["true_rul"] > 40) & (eval_df["true_rul"] <= 80)]),
        ("Nominal Cruise (RUL > 80)", eval_df[eval_df["true_rul"] > 80]),
    ]
    for name, r_df in regimes:
        if len(r_df) > 0:
            r_rmse = np.sqrt(np.mean(r_df["error"] ** 2))
            r_mae = np.mean(r_df["abs_error"])
            print(f"{name:38s} | Engines: {len(r_df):2d} | MAE: {r_mae:5.2f} | RMSE: {r_rmse:5.2f}")

    # Save detailed CSV
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"test_validation_{subset}.csv")
    eval_df.to_csv(out_csv, index=False)
    print(f"\nDetailed per-engine validation report saved to: {out_csv}")
    print("=" * 80)

    return eval_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="FD001", help="C-MAPSS subset identifier")
    args = parser.parse_args()
    evaluate_subset_test_predictions(subset=args.subset)
