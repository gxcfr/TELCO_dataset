"""
preprocessing.py
================
Module for Feature Transformation, Encoding, and Pipeline Construction.

Implements:
- Identification of numeric and categorical feature sets
- Standardized target encoding (Yes -> 1, No -> 0)
- Scikit-Learn ColumnTransformer combining:
    * StandardScaler (essential for regularized Logistic Regression gradient convergence)
    * OneHotEncoder (handle_unknown='ignore' for robust production inference)
    * SimpleImputer (defensive against missing entries at inference time)
- Stratified Train-Test Splitting (80:20) ensuring identical class distribution
"""

import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_types(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical columns, excluding target column.
    Compatible with Pandas 2 and 3 StringDtypes.
    """
    feature_df = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. Imputes and scales numerical features using StandardScaler.
       Scaling is vital for Logistic Regression because:
       - Regularization penalties (L2/L1) penalize weights uniformly; unscaled features
         with large ranges (like TotalCharges ~8000) would unfairly dominate gradients.
       - Solvers (lbfgs, sag) converge significantly faster and reliably with scaled inputs.
    2. Imputes and encodes categorical features using OneHotEncoder.
       - Logistic regression requires numerical design matrices (cannot accept strings).
       - handle_unknown='ignore' ensures robust out-of-sample prediction without runtime crashes.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified 80:20 train-test split.
    Encodes target: 'Yes' -> 1, 'No' -> 0.
    Stratification ensures the 26.5% churn minority class is proportionally preserved
    in both training and test partitions.
    """
    X = df.drop(columns=[target_col])
    
    # Encode target cleanly regardless of underlying pandas string representation
    target_series = df[target_col]
    if target_series.astype(str).str.strip().isin(["Yes", "No"]).any():
        y = target_series.astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)
    else:
        y = target_series.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
