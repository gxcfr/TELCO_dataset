"""
run_pipeline.py
===============
End-to-end execution runner that validates all modules, trains models,
computes metrics, generates plots, and serializes artifacts.
"""

import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaning import load_raw_data, inspect_raw_quality, clean_data, detect_outliers_iqr
from src.feature_engineering import engineer_all_features
from src.model_training import run_experiments, save_pipeline
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_metric_comparison,
    plot_actual_vs_predicted,
    calculate_coefficients_and_odds_ratios,
    plot_top_coefficients,
)


def main():
    print("=" * 60)
    print("TELCO CHURN PREDICTION – END-TO-END PIPELINE EXECUTION")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    plots_dir = os.path.join(base_dir, "outputs", "plots")
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    models_dir = os.path.join(base_dir, "outputs", "models")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/7] Loading raw data...")
    df_raw = load_raw_data(raw_path)
    print(f"Raw data shape: {df_raw.shape}")

    # 2. Audit & Clean Data
    print("\n[Step 2/7] Auditing and cleaning data...")
    quality_audit = inspect_raw_quality(df_raw)
    print(f"Blank TotalCharges detected: {quality_audit['blank_total_charges']}")
    print(f"Duplicate rows detected: {quality_audit['duplicate_rows']}")

    df_clean = clean_data(df_raw, drop_id=True)
    df_clean.to_csv(os.path.join(processed_dir, "cleaned_telco_churn.csv"), index=False)
    print(f"Cleaned dataset saved: {df_clean.shape}")

    outlier_audit = detect_outliers_iqr(df_clean)
    with open(os.path.join(metrics_dir, "data_quality_audit.json"), "w") as f:
        json.dump({"raw_quality": quality_audit, "outlier_audit": outlier_audit}, f, indent=2)

    # 3. Feature Engineering
    print("\n[Step 3/7] Performing domain feature engineering...")
    df_engineered = engineer_all_features(df_clean)
    df_engineered.to_csv(os.path.join(processed_dir, "engineered_telco_churn.csv"), index=False)
    print(f"Engineered dataset shape: {df_engineered.shape}")
    new_cols = [c for c in df_engineered.columns if c not in df_clean.columns]
    print(f"Newly created features ({len(new_cols)}): {new_cols}")

    # 4. Model Training & Experiments
    print("\n[Step 4/7] Training Logistic Regression models across experiments...")
    exp_results = run_experiments(df_clean, df_engineered, test_size=0.2, random_state=42)

    pipeline_base = exp_results["pipeline_base"]
    pipeline_eng = exp_results["pipeline_eng"]
    pipeline_eng_bal = exp_results["pipeline_eng_bal"]
    splits = exp_results["splits"]

    # Save models
    save_pipeline(pipeline_base, os.path.join(models_dir, "logistic_regression_baseline.joblib"))
    save_pipeline(pipeline_eng, os.path.join(models_dir, "logistic_regression_engineered.joblib"))
    save_pipeline(pipeline_eng_bal, os.path.join(models_dir, "logistic_regression_engineered_balanced.joblib"))

    # 5. Model Evaluation
    print("\n[Step 5/7] Evaluating models on identical test partition...")
    res_base = evaluate_model(pipeline_base, splits["X_test_base"], splits["y_test"], "Baseline Model")
    res_eng = evaluate_model(pipeline_eng, splits["X_test_eng"], splits["y_test"], "Engineered Model")
    res_bal = evaluate_model(pipeline_eng_bal, splits["X_test_eng"], splits["y_test"], "Engineered (Balanced)")

    metrics_table = pd.DataFrame([
        res_base["metrics"],
        res_eng["metrics"],
        res_bal["metrics"],
    ])[["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]

    print("\nModel Evaluation Comparison Table:")
    print(metrics_table.to_string(index=False))
    metrics_table.to_csv(os.path.join(metrics_dir, "model_comparison_metrics.csv"), index=False)

    # 6. Generate Evaluation Plots
    print("\n[Step 6/7] Generating and saving evaluation charts...")
    plot_confusion_matrix(res_base["metrics"]["Confusion_Matrix"], "Baseline Model", os.path.join(plots_dir, "confusion_matrix_baseline.png"))
    plot_confusion_matrix(res_eng["metrics"]["Confusion_Matrix"], "Engineered Model", os.path.join(plots_dir, "confusion_matrix_engineered.png"))
    plot_confusion_matrix(res_bal["metrics"]["Confusion_Matrix"], "Engineered (Balanced)", os.path.join(plots_dir, "confusion_matrix_engineered_balanced.png"))

    plot_roc_curves([res_base, res_eng, res_bal], os.path.join(plots_dir, "roc_curve_comparison.png"))
    plot_precision_recall_curves([res_base, res_eng, res_bal], os.path.join(plots_dir, "precision_recall_curve_comparison.png"))
    plot_metric_comparison(metrics_table, os.path.join(plots_dir, "model_metric_comparison_barchart.png"))

    plot_actual_vs_predicted(res_eng["y_true"], res_eng["y_pred"], "Engineered Model", os.path.join(plots_dir, "actual_vs_predicted_engineered.png"))

    # 7. Model Interpretation & Odds Ratios
    print("\n[Step 7/7] Computing model coefficients and odds ratios...")
    df_coef_eng = calculate_coefficients_and_odds_ratios(pipeline_eng)
    df_coef_eng.to_csv(os.path.join(metrics_dir, "engineered_model_coefficients_and_odds.csv"), index=False)

    df_coef_base = calculate_coefficients_and_odds_ratios(pipeline_base)
    df_coef_base.to_csv(os.path.join(metrics_dir, "baseline_model_coefficients_and_odds.csv"), index=False)

    plot_top_coefficients(df_coef_eng, top_n=12, model_name="Engineered Model", save_path=os.path.join(plots_dir, "top_coefficients_engineered.png"))

    print("\nTop Predictors of Churn Odds Increase:")
    print(df_coef_eng[df_coef_eng["Coefficient"] > 0].head(5)[["Feature", "Coefficient", "Odds_Ratio"]].to_string(index=False))

    print("\nTop Predictors of Churn Odds Reduction (Retention Drivers):")
    print(df_coef_eng[df_coef_eng["Coefficient"] < 0].head(5)[["Feature", "Coefficient", "Odds_Ratio"]].to_string(index=False))

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()
