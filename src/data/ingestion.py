"""
Data Ingestion Module for NASA C-MAPSS dataset.
Handles loading raw text files with proper column schemas and type conversions.
"""

import os
from typing import Optional, Tuple
import pandas as pd

COLUMN_NAMES = [
    "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

def get_data_filepath(data_dir: str, subset: str, split: str) -> str:
    """Resolve file path for a given subset and split."""
    filename = f"{split}_{subset}.txt"
    # Check in data_dir directly, or in data_dir/raw
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join(data_dir, "raw", filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not locate dataset file: {filename} in {candidates}")

def load_cmapss_data(data_dir: str = "data/raw", subset: str = "FD001", split: str = "train") -> pd.DataFrame:
    """
    Load raw space-delimited C-MAPSS dataset file with standardized schema.
    
    Args:
        data_dir: Directory containing raw data files.
        subset: C-MAPSS subset identifier ('FD001', 'FD002', etc.).
        split: 'train' or 'test'.
        
    Returns:
        DataFrame containing parsed telemetry.
    """
    filepath = get_data_filepath(data_dir, subset, split)
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        dtype={"unit": int, "cycle": int}
    )
    return df

def load_test_rul(data_dir: str = "data/raw", subset: str = "FD001") -> pd.DataFrame:
    """
    Load ground-truth Remaining Useful Life (RUL) for test engines at final cycle.
    
    Returns:
        DataFrame with columns ['unit', 'true_rul'].
    """
    filename = f"RUL_{subset}.txt"
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join(data_dir, "raw", filename),
        filename
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break
            
    if filepath is None:
        raise FileNotFoundError(f"Could not locate RUL file: {filename}")
        
    ruls = pd.read_csv(filepath, sep=r"\s+", header=None, names=["true_rul"])
    ruls["unit"] = range(1, len(ruls) + 1)
    return ruls[["unit", "true_rul"]]
