"""
Data Validation Module for NASA C-MAPSS telemetry.
Enforces schema contracts, checks for missing data, and verifies monotonic cycle sequences.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.data.ingestion import COLUMN_NAMES

class DataValidationError(Exception):
    """Raised when dataset integrity or schema checks fail."""
    pass

def validate_schema(df: pd.DataFrame) -> bool:
    """Check that all expected columns exist and are present in the DataFrame."""
    missing = set(COLUMN_NAMES) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected columns in telemetry DataFrame: {missing}")
    return True

def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Verify that there are no NaN or null values in the dataset."""
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        bad_cols = {k: v for k, v in null_counts.items() if v > 0}
        raise DataValidationError(f"Found {total_nulls} null values across columns: {bad_cols}")
    return null_counts

def validate_engine_cycles(df: pd.DataFrame) -> bool:
    """
    Ensure that operating cycles for every engine start at 1,
    are strictly increasing, and have no duplicate cycles.
    """
    for unit_id, group in df.groupby("unit"):
        cycles = group["cycle"].values
        if len(cycles) == 0:
            raise DataValidationError(f"Unit {unit_id} contains no cycle records.")
        if cycles[0] != 1:
            raise DataValidationError(f"Unit {unit_id} starts at cycle {cycles[0]}, expected cycle 1.")
        diffs = np.diff(cycles)
        if not np.all(diffs == 1):
            raise DataValidationError(f"Unit {unit_id} has non-contiguous or non-increasing cycles: {cycles}")
    return True

def identify_constant_sensors(df: pd.DataFrame, variance_threshold: float = 1e-4) -> Tuple[List[str], List[str]]:
    """
    Separate sensors into invariant (near-zero variance) and active degrading sensors.
    
    Returns:
        (active_sensors, constant_sensors)
    """
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    stds = df[sensor_cols].std()
    
    constant_sensors = stds[stds < variance_threshold].index.tolist()
    active_sensors = [s for s in sensor_cols if s not in constant_sensors]
    
    return active_sensors, constant_sensors

def run_full_validation(df: pd.DataFrame) -> Dict[str, any]:
    """Execute complete validation suite on input dataframe."""
    validate_schema(df)
    check_missing_values(df)
    validate_engine_cycles(df)
    active_s, const_s = identify_constant_sensors(df)
    return {
        "status": "VALID",
        "total_rows": len(df),
        "total_units": df["unit"].nunique(),
        "active_sensors": active_s,
        "constant_sensors": const_s
    }
