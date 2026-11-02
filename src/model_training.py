"""
model_training.py
=================
Module for Training Logistic Regression Models and Comparative Experiments.

Provides:
- Experiment A: Baseline Model (Original cleaned features)
- Experiment B: Feature Engineered Model (Enriched domain features)
- Support for class_weight tuning ('balanced' vs default) for recall optimization
- Model serialization via joblib for seamless Streamlit inference
"""

import os
import joblib
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.preprocessing import get_feature_types, build_preprocessor, split_data


def build_pipeline(
    preprocessor: ColumnTransformer,
    C: float = 1.0,
    penalty: str = "l2",
    class_weight: Any = None,
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn Pipeline combining the ColumnTransformer
    preprocessor with the LogisticRegression classifier.
    Ensures all transformations are fitted strictly on training data without leakage.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])
    return pipeline


def train_model(
    X_train,
    y_train,
    preprocessor: ColumnTransformer,
    class_weight: Any = None,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Builds and fits a Logistic Regression pipeline on the training partition.
    """
    pipeline = build_pipeline(
        preprocessor=preprocessor,
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def run_experiments(
    df_clean,
    df_engineered,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes controlled experiments:
    - Experiment A: Baseline Model
    - Experiment B: Feature Engineered Model
    - Experiment B (Balanced): Feature Engineered with class_weight='balanced'
    
    Guarantees strict experimental rigor by using identical train/test indices.
    """
    # 1. Split baseline data
    X_train_base, X_test_base, y_train, y_test = split_data(
        df_clean, target_col="Churn", test_size=test_size, random_state=random_state
    )

    # 2. Split engineered data using identical target vector
    X_train_eng = df_engineered.loc[X_train_base.index].drop(columns=["Churn"])
    X_test_eng = df_engineered.loc[X_test_base.index].drop(columns=["Churn"])

    # 3. Build preprocessors
    num_base, cat_base = get_feature_types(df_clean, target_col="Churn")
    preproc_base = build_preprocessor(num_base, cat_base)

    num_eng, cat_eng = get_feature_types(df_engineered, target_col="Churn")
    preproc_eng = build_preprocessor(num_eng, cat_eng)

    # 4. Train Experiment A: Baseline
    pipeline_base = train_model(
        X_train_base, y_train, preprocessor=preproc_base, class_weight=None, random_state=random_state
    )

    # 5. Train Experiment B: Engineered (Standard class weights)
    pipeline_eng = train_model(
        X_train_eng, y_train, preprocessor=preproc_eng, class_weight=None, random_state=random_state
    )

    # 6. Train Experiment B: Engineered (Balanced class weights for recall boost)
    pipeline_eng_bal = train_model(
        X_train_eng, y_train, preprocessor=preproc_eng, class_weight="balanced", random_state=random_state
    )

    results = {
        "pipeline_base": pipeline_base,
        "pipeline_eng": pipeline_eng,
        "pipeline_eng_bal": pipeline_eng_bal,
        "splits": {
            "X_train_base": X_train_base,
            "X_test_base": X_test_base,
            "X_train_eng": X_train_eng,
            "X_test_eng": X_test_eng,
            "y_train": y_train,
            "y_test": y_test,
        },
        "feature_types": {
            "num_base": num_base,
            "cat_base": cat_base,
            "num_eng": num_eng,
            "cat_eng": cat_eng,
        }
    }

    return results


def save_pipeline(pipeline: Pipeline, filepath: str):
    """
    Serializes trained pipeline to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to: {filepath}")


def load_pipeline(filepath: str) -> Pipeline:
    """
    Loads serialized pipeline from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline file not found: {filepath}")
    return joblib.load(filepath)
