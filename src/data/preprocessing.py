"""
Preprocessing Module for C-MAPSS dataset.
Calculates ground truth RUL, applies piecewise linear clipping,
and performs engine-level train/validation splitting without data leakage.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

def calculate_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for run-to-failure training data.
    RUL at cycle t = max_cycle - t.
    """
    df_out = df.copy()
    max_cycles = df_out.groupby("unit")["cycle"].transform("max")
    df_out["RUL"] = max_cycles - df_out["cycle"]
    return df_out

def apply_rul_cap(df: pd.DataFrame, cap: Optional[int] = 125) -> pd.DataFrame:
    """
    Apply piecewise linear RUL clipping.
    In early engine life, degradation is not yet active, so capping RUL
    at e.g. 125 cycles aligns model training with the actual onset of degradation.
    """
    df_out = df.copy()
    if cap is not None and "RUL" in df_out.columns:
        df_out["RUL_capped"] = df_out["RUL"].clip(upper=cap)
    return df_out

def split_engines_by_unit(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Split dataset into training and validation sets strictly by engine unit ID.
    Guarantees that all observations of any single engine remain either in
    train or in validation, preventing severe temporal data leakage.
    
    Returns:
        (train_df, val_df, train_units, val_units)
    """
    unique_units = np.array(sorted(df["unit"].unique()))
    np.random.seed(random_state)
    shuffled = np.random.permutation(unique_units)
    
    n_train = int(len(unique_units) * train_ratio)
    train_units = sorted(shuffled[:n_train].tolist())
    val_units = sorted(shuffled[n_train:].tolist())
    
    train_df = df[df["unit"].isin(train_units)].copy().reset_index(drop=True)
    val_df = df[df["unit"].isin(val_units)].copy().reset_index(drop=True)
    
    return train_df, val_df, train_units, val_units
