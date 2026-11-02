"""
feature_engineering.py
======================
Module for Domain-Specific Feature Engineering on Telco Customer Data.

Constructs high-signal behavioral and contractual features:
- TenureGroup (Binned tenure categories)
- TotalServices (Aggregate service count)
- HasOnlineSecurity (High-retention security flag)
- HasOnlineBackup (Data protection flag)
- HasTechSupport (Technical assistance flag)
- HasMultipleServices (Multi-product relationship indicator)
- IsLongTermContract (Commitment-based retention flag)
- IsAutomaticPayment (Frictionless billing flag)
- AverageMonthlySpend (Effective monthly spend accounting for tenure=0)
"""

import pandas as pd
import numpy as np


def engineer_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins customer tenure into business-meaningful life-cycle stages:
    - New customer: 0 to 12 months (highest risk of early churn)
    - Short-term: 13 to 24 months
    - Medium-term: 25 to 48 months
    - Long-term: > 48 months (loyal, deeply embedded customers)
    """
    df = df.copy()
    bins = [-1, 12, 24, 48, np.inf]
    labels = ["New customer", "Short-term", "Medium-term", "Long-term"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    df["TenureGroup"] = df["TenureGroup"].astype(str)
    return df


def engineer_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers service engagement flags and cumulative service count.
    Higher service stickiness typically suppresses customer churn.
    """
    df = df.copy()

    # Individual key value-added service indicators (1 if 'Yes', 0 otherwise)
    df["HasOnlineSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)
    df["HasOnlineBackup"] = (df["OnlineBackup"] == "Yes").astype(int)
    df["HasTechSupport"] = (df["TechSupport"] == "Yes").astype(int)

    # List of all potential add-on / core services
    service_cols = [
        ("PhoneService", "Yes"),
        ("MultipleLines", "Yes"),
        ("OnlineSecurity", "Yes"),
        ("OnlineBackup", "Yes"),
        ("DeviceProtection", "Yes"),
        ("TechSupport", "Yes"),
        ("StreamingTV", "Yes"),
        ("StreamingMovies", "Yes"),
    ]

    total_services = pd.Series(0, index=df.index)
    for col, target_val in service_cols:
        if col in df.columns:
            total_services += (df[col] == target_val).astype(int)

    # If InternetService is DSL or Fiber optic, count internet connection as a service
    if "InternetService" in df.columns:
        total_services += (df["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)

    df["TotalServices"] = total_services.astype(int)

    # Customer has multiple bundled services (> 1 service)
    df["HasMultipleServices"] = (df["TotalServices"] > 1).astype(int)

    return df


def engineer_contract_and_billing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers contract horizon and automated payment flags.
    Contractual lock-in and seamless auto-debit are strong retention drivers.
    """
    df = df.copy()

    # IsLongTermContract: 1 for 1-year or 2-year contracts, 0 for Month-to-month
    if "Contract" in df.columns:
        df["IsLongTermContract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    else:
        df["IsLongTermContract"] = 0

    # IsAutomaticPayment: 1 for automatic bank transfer or credit card
    auto_payments = ["Bank transfer (automatic)", "Credit card (automatic)"]
    if "PaymentMethod" in df.columns:
        df["IsAutomaticPayment"] = df["PaymentMethod"].isin(auto_payments).astype(int)
    else:
        df["IsAutomaticPayment"] = 0

    return df


def engineer_spend_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives AverageMonthlySpend from TotalCharges and tenure.
    Safely handles tenure == 0 by substituting current MonthlyCharges to prevent division by zero.
    """
    df = df.copy()

    # If tenure > 0, calculate TotalCharges / tenure; otherwise fallback to MonthlyCharges
    safe_spend = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    df["AverageMonthlySpend"] = np.round(safe_spend, 2)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline orchestrating all feature engineering steps.
    Strictly avoids target leakage.
    """
    df_eng = df.copy()
    df_eng = engineer_tenure_group(df_eng)
    df_eng = engineer_service_features(df_eng)
    df_eng = engineer_contract_and_billing(df_eng)
    df_eng = engineer_spend_metrics(df_eng)
    return df_eng
