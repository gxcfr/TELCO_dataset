"""
evaluation.py
=============
Module for Machine Learning Model Evaluation, Visualizations, and Interpretation.

Provides:
- Comprehensive performance metric calculations (Accuracy, Precision, Recall, F1, ROC-AUC)
- Publication-quality visualization generation (Confusion Matrices, ROC Curves, PR Curves, Metric Comparisons)
- Logistic Regression feature interpretation (Coefficients, Odds Ratios, Business Drivers)
- Exporting plots to outputs/plots/ and metrics to outputs/metrics/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Logistic Regression"
) -> dict:
    """
    Computes standard classification evaluation metrics:
    Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    Returns metrics dict and probability predictions.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "Model": model_name,
        "Accuracy": float(round(acc, 4)),
        "Precision": float(round(prec, 4)),
        "Recall": float(round(rec, 4)),
        "F1-score": float(round(f1, 4)),
        "ROC-AUC": float(round(roc_auc, 4)),
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": report_dict,
        "Classification_Report_Text": report_text,
    }

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_test.values,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: str = None
) -> plt.Figure:
    """
    Plots an annotated confusion matrix with clear class labels.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Class", fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_roc_curves(
    eval_results_list: list,
    save_path: str = None
) -> plt.Figure:
    """
    Plots overlaid ROC curves for comparative model evaluation.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#2563EB", "#10B981", "#F59E0B"]
    for i, res in enumerate(eval_results_list):
        model_name = res["metrics"]["Model"]
        roc_auc = res["metrics"]["ROC-AUC"]
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", color=color, lw=2.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.500)", alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_precision_recall_curves(
    eval_results_list: list,
    save_path: str = None
) -> plt.Figure:
    """
    Plots overlaid Precision-Recall curves.
    PR curves are especially informative for imbalanced churn datasets.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#2563EB", "#10B981", "#F59E0B"]
    for i, res in enumerate(eval_results_list):
        model_name = res["metrics"]["Model"]
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ap = average_precision_score(res["y_true"], res["y_prob"])
        color = colors[i % len(colors)]
        ax.plot(rec, prec, label=f"{model_name} (AP = {ap:.3f})", color=color, lw=2.2)

    no_skill = np.mean(eval_results_list[0]["y_true"])
    ax.plot([0, 1], [no_skill, no_skill], "k--", lw=1.5, label=f"Baseline Proportion ({no_skill:.2f})", alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Coverage of Churners)", fontsize=11)
    ax.set_ylabel("Precision (Accuracy of Churn Flags)", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Produces a grouped bar chart comparing key evaluation metrics across models.
    """
    melted = metrics_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    melted.columns = ["Model", "Metric", "Score"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#3B82F6", "#10B981", "#F59E0B"],
        ax=ax
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Evaluation Metric Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bar heights
    for p in ax.patches:
        h = p.get_height()
        if h > 0.01:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None
) -> plt.Figure:
    """
    Plots comparative distributions of actual vs predicted class outcomes.
    """
    df_counts = pd.DataFrame({
        "Actual": pd.Series(y_true).map({0: "Retained", 1: "Churned"}),
        "Predicted": pd.Series(y_pred).map({0: "Retained", 1: "Churned"}),
    })
    counts = pd.melt(df_counts, value_vars=["Actual", "Predicted"], var_name="Type", value_name="Outcome")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(
        data=counts,
        x="Outcome",
        hue="Type",
        palette={"Actual": "#64748B", "Predicted": "#2563EB"},
        ax=ax
    )
    ax.set_title(f"Actual vs Predicted Class Distribution – {model_name}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Customer Status", fontsize=11)
    ax.set_ylabel("Count of Customers", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def extract_feature_names(preprocessor: Pipeline) -> list:
    """
    Safely retrieves transformed feature names from a Scikit-Learn ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if manual extraction needed
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            elif hasattr(transformer, "named_steps") and hasattr(transformer.named_steps.get("encoder"), "get_feature_names_out"):
                feature_names.extend(transformer.named_steps["encoder"].get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        return feature_names


def calculate_coefficients_and_odds_ratios(
    pipeline: Pipeline,
    clean_feature_names: bool = True
) -> pd.DataFrame:
    """
    Extracts Logistic Regression coefficients and computes Odds Ratios:
    Odds Ratio = exp(beta)
    
    Interpretation Guidelines (Non-Causal):
    - beta > 0 (OR > 1): Feature is positively associated with higher odds of churn.
    - beta < 0 (OR < 1): Feature is negatively associated with churn (protective / retention association).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    raw_names = extract_feature_names(preprocessor)
    coefs = model.coef_[0]

    # Clean prefixes added by ColumnTransformer (e.g. num__, cat__)
    if clean_feature_names:
        cleaned_names = [name.replace("num__", "").replace("cat__", "") for name in raw_names]
    else:
        cleaned_names = raw_names

    df_coef = pd.DataFrame({
        "Feature": cleaned_names,
        "Coefficient": np.round(coefs, 4),
        "Odds_Ratio": np.round(np.exp(coefs), 4),
        "Impact": ["Increases Churn Odds" if c > 0 else "Decreases Churn Odds" for c in coefs]
    })

    df_coef["Abs_Coefficient"] = df_coef["Coefficient"].abs()
    df_coef = df_coef.sort_values(by="Abs_Coefficient", ascending=False).drop(columns=["Abs_Coefficient"])
    return df_coef


def plot_top_coefficients(
    df_coef: pd.DataFrame,
    top_n: int = 10,
    model_name: str = "Logistic Regression",
    save_path: str = None
) -> plt.Figure:
    """
    Plots horizontal bar chart of the strongest positive and negative predictors.
    """
    top_pos = df_coef[df_coef["Coefficient"] > 0].head(top_n // 2)
    top_neg = df_coef[df_coef["Coefficient"] < 0].head(top_n // 2)
    df_top = pd.concat([top_pos, top_neg]).sort_values(by="Coefficient", ascending=True)

    colors = ["#10B981" if c < 0 else "#EF4444" for c in df_top["Coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(df_top["Feature"], df_top["Coefficient"], color=colors, height=0.65)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top Churn Predictors (Coefficients) – {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Logistic Regression Coefficient (Log-Odds)", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (0.02 if w >= 0 else -0.02)
        ha = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2.0, f"{w:+.2f}", ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
