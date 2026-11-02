"""
data_cleaning.py
================
Module for Telco Customer Churn Data Cleaning and Quality Assurance.

Provides modular functions for:
- Loading raw CSV data gracefully
- Handling blank strings and type conversion in TotalCharges
- Verifying missing values and duplicate records
- Handling outlier inspection without destructive truncation
- Removing non-predictive identifiers (customerID)
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset from the given path.
    Raises FileNotFoundError with helpful instructions if missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is placed in 'data/raw/'."
        )
    df = pd.read_csv(filepath)
    return df


def inspect_raw_quality(df: pd.DataFrame) -> dict:
    """
    Performs data quality audit on raw dataframe.
    Returns audit summary dictionary.
    """
    blank_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    duplicates_count = df.duplicated().sum()
    missing_counts = df.isnull().sum().to_dict()

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "blank_total_charges": int(blank_total_charges),
        "duplicate_rows": int(duplicates_count),
        "missing_values_by_col": missing_counts,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the TotalCharges column:
    - Identifies blank string values (' ')
    - Explains domain context: blank TotalCharges correspond strictly to tenure=0 (new customers)
    - Replaces blanks with 0.0 (realistic accumulated billing for 0 tenure)
    - Casts column to float64
    """
    df = df.copy()

    # Convert spaces/empty strings to NaN first
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)

    # For customers with tenure == 0, TotalCharges is logically 0.0
    # For any remaining NaNs, fill with 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Cast to numeric float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    return df


def clean_data(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Comprehensive cleaning pipeline:
    1. Clean and cast TotalCharges
    2. Check and remove duplicates
    3. Drop customerID (non-predictive unique identifier)
    4. Ensure correct categorical dtypes
    """
    df_clean = df.copy()

    # 1. Clean TotalCharges
    df_clean = clean_total_charges(df_clean)

    # 2. Check and remove duplicates
    df_clean = df_clean.drop_duplicates()

    # 3. Drop customerID if requested
    if drop_id and "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # 4. Standardize SeniorCitizen as categorical string for consistent encoding
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype(str)

    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> dict:
    """
    Calculates IQR-based outlier boundaries for numerical features.
    Does not delete them blindly; telecom charges naturally span a wide
    valid operational range.
    """
    if columns is None:
        columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    outlier_report = {}
    for col in columns:
        if col in df.columns:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_report[col] = {
                "Q1 (25%)": float(round(q25, 2)),
                "Q3 (75%)": float(round(q75, 2)),
                "IQR": float(round(iqr, 2)),
                "Lower Bound": float(round(lower_bound, 2)),
                "Upper Bound": float(round(upper_bound, 2)),
                "Outlier Count": outlier_count,
                "Outlier Percentage": float(round(outlier_count / len(df) * 100, 2)),
            }
    return outlier_report
