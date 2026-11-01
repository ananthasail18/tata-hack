"""
Evaluation metrics module for NASA C-MAPSS RUL regression.
Implements RMSE, MAE, and the official NASA PHM 2008 Asymmetric Scoring Function.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official NASA PHM 2008 asymmetric scoring function.
    Penalizes late predictions (overestimating RUL) significantly more heavily
    than early predictions (underestimating RUL), reflecting operational safety risk.
    
    d_i = y_pred - y_true
    score = sum( exp(-d/13) - 1 for d < 0 ) + sum( exp(d/10) - 1 for d >= 0 )
    """
    d = np.array(y_pred) - np.array(y_true)
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standardized regression evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nasa_score = float(nasa_scoring_function(y_true, y_pred))
    
    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "nasa_score": round(nasa_score, 2)
    }
