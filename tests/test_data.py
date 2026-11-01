"""
Unit tests for Data Ingestion, Validation, and Preprocessing.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_cmapss_data, load_test_rul, COLUMN_NAMES
from src.data.validation import (
    validate_schema,
    check_missing_values,
    validate_engine_cycles,
    identify_constant_sensors,
    DataValidationError
)
from src.data.preprocessing import (
    calculate_training_rul,
    apply_rul_cap,
    split_engines_by_unit
)

def test_data_ingestion():
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    assert len(df) == 20631, f"Expected 20,631 rows, got {len(df)}"
    assert list(df.columns) == COLUMN_NAMES
    assert df["unit"].nunique() == 100

def test_test_rul_ingestion():
    ruls = load_test_rul(data_dir="data/raw", subset="FD001")
    assert len(ruls) == 100, f"Expected 100 test engines, got {len(ruls)}"
    assert "true_rul" in ruls.columns

def test_data_validation():
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    assert validate_schema(df) is True
    
    nulls = check_missing_values(df)
    assert sum(nulls.values()) == 0
    
    assert validate_engine_cycles(df) is True

def test_constant_sensors_identification():
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    # In FD001, sensors 1, 5, 10, 16, 18, 19 have variance 0.0
    # Sensor 6 has tiny variance ~0.0014 (quasi-constant)
    active, const = identify_constant_sensors(df, variance_threshold=0.01)
    assert len(active) == 14
    assert len(const) == 7
    assert "sensor_1" in const
    assert "sensor_4" in active

def test_rul_calculation():
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    df_rul = calculate_training_rul(df)
    
    # Engine 1 operates for 192 cycles
    u1 = df_rul[df_rul["unit"] == 1]
    assert u1.iloc[0]["RUL"] == 191  # At cycle 1: 192 - 1 = 191
    assert u1.iloc[-1]["RUL"] == 0   # Final cycle: failure, RUL = 0

def test_engine_wise_split_no_leakage():
    df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    train_df, val_df, train_units, val_units = split_engines_by_unit(df, train_ratio=0.8, random_state=42)
    
    # Verify no intersection between train and val units!
    intersection = set(train_units).intersection(set(val_units))
    assert len(intersection) == 0, f"Data leakage detected! Shared units: {intersection}"
    assert len(train_units) == 80
    assert len(val_units) == 20
