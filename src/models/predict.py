"""
Inference Module for NASA C-MAPSS RUL Prediction.
Loads serialized models and provides unified prediction methods for both
single-cycle telemetry and multi-engine batch datasets.
"""

import os
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import torch

from src.features.engineering import FeaturePipeline
from src.models.train import TurbofanLSTM, predict_lstm

class RULPredictor:
    """
    Unified predictor wrapper for C-MAPSS models.
    Handles feature preprocessing, sequence formatting (if LSTM), and prediction clipping.
    """
    def __init__(self, model: any, pipeline: FeaturePipeline, model_type: str = "xgboost", rul_cap: float = 125.0):
        self.model = model
        self.pipeline = pipeline
        self.model_type = model_type.lower()
        self.rul_cap = rul_cap

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Generate RUL predictions from raw telemetry dataframe.
        """
        X_scaled, _, feat_df = self.pipeline.transform(df_raw)
        
        if self.model_type == "lstm":
            preds = predict_lstm(self.model, X_scaled, feat_df["unit"].values, seq_len=30)
        else:
            preds = self.model.predict(X_scaled)
            
        # Ensure RUL predictions cannot be physically negative
        preds = np.clip(preds, 0.0, None)
        return np.round(preds, 2)

    def predict_final_cycles(self, df_test_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predict RUL for the final observed operational cycle of each test engine.
        Returns a DataFrame with columns ['unit', 'final_cycle', 'predicted_rul'].
        """
        preds_all = self.predict(df_test_raw)
        
        # Associate with metadata
        res_df = df_test_raw[["unit", "cycle"]].copy()
        res_df["predicted_rul"] = preds_all
        
        # Extract last row per unit
        idx_last = res_df.groupby("unit")["cycle"].idxmax()
        final_preds = res_df.loc[idx_last].reset_index(drop=True)
        final_preds.rename(columns={"cycle": "final_cycle"}, inplace=True)
        return final_preds

    def save(self, filepath: str) -> None:
        """Serialize predictor wrapper to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.model_type == "lstm":
            # For PyTorch model, save state dict alongside metadata
            save_dict = {
                "model_type": self.model_type,
                "pipeline": self.pipeline,
                "rul_cap": self.rul_cap,
                "state_dict": self.model.state_dict(),
                "input_dim": len(self.pipeline.feature_names)
            }
            torch.save(save_dict, filepath)
        else:
            joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "RULPredictor":
        """Load serialized predictor wrapper."""
        if filepath.endswith(".pt") or filepath.endswith(".pth"):
            ckpt = torch.load(filepath, map_location="cpu")
            model = TurbofanLSTM(input_dim=ckpt["input_dim"], hidden_dim=64, num_layers=2)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            return cls(model=model, pipeline=ckpt["pipeline"], model_type="lstm", rul_cap=ckpt["rul_cap"])
        else:
            return joblib.load(filepath)
