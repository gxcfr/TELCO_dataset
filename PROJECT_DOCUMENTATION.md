# Telco Customer Churn Prediction: Complete End-to-End Project Documentation

**Project Repository:** [https://github.com/gxcfr/TELCO_dataset](https://github.com/gxcfr/TELCO_dataset)  
**Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Primary Algorithm:** Logistic Regression (Baseline, Feature-Engineered, and Recall-Optimized Specifications)  
**Core Frameworks:** Python 3.11, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook, Streamlit  

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Dataset Overview & Target Definition](#2-dataset-overview--target-definition)
3. [Full Project Architecture & Directory Layout](#3-full-project-architecture--directory-layout)
4. [Phase 1: Exploratory Data Analysis (EDA) Findings](#4-phase-1-exploratory-data-analysis-eda-findings)
5. [Phase 2: Data Cleaning & Quality Assurance](#5-phase-2-data-cleaning--quality-assurance)
6. [Phase 3: Domain-Specific Feature Engineering](#6-phase-3-domain-specific-feature-engineering)
7. [Phase 4: Preprocessing & Leak-Free Architecture](#7-phase-4-preprocessing--leak-free-architecture)
8. [Phase 5: Logistic Regression Mathematical Formulation](#8-phase-5-logistic-regression-mathematical-formulation)
9. [Phase 6: Model Evaluation & Experimental Comparison](#9-phase-6-model-evaluation--experimental-comparison)
10. [Phase 7: Model Interpretability & Odds Ratios](#10-phase-7-model-interpretability--odds-ratios)
11. [Phase 8: Step-by-Step Instructions to Launch Notebook & Dashboard](#11-phase-8-step-by-step-instructions-to-launch-notebook--dashboard)
12. [Academic Viva Defense Questions & Answers](#12-academic-viva-defense-questions--answers)
13. [Limitations & Future Enhancements](#13-limitations--future-enhancements)

---

## 1. Executive Summary & Problem Statement

In the telecommunications industry, customer acquisition costs typically exceed customer retention costs by **5 to 7 times**. When a customer terminates their service (churns), the business suffers:
- Loss of recurring monthly subscription revenue (MRR).
- Sunk customer acquisition costs (CAC).
- Diminished Customer Lifetime Value (CLV).

The objective of this project is to build an **auditable, leak-free, academically defensible Machine Learning pipeline** that identifies customers at risk of churning, uncovers the primary behavioral and contractual drivers of churn, and provides interactive real-time inference tools for retention specialists.

Unlike black-box models, this implementation focuses on **Logistic Regression**, providing mathematical interpretability via **Odds Ratios** ($OR = e^\beta$) while handling real-world data imperfections.

---

## 2. Dataset Overview & Target Definition

- **Dataset Source:** IBM Telco Customer Churn (blastchar on Kaggle / IBM Sample Data)
- **File Name:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Located in `data/raw/`)
- **Total Records:** 7,043 customers
- **Total Features:** 21 columns (20 predictor variables + 1 target variable)
- **Target Variable (`Churn`):**
  - **`No` (0):** Customer stayed (5,174 customers, **73.46%**).
  - **`Yes` (1):** Customer churned (1,869 customers, **26.54%**).
  - **Class Ratio:** ~2.77:1 imbalance.

### Feature Categorization
| Category | Columns |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account & Billing** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Telephony Services** | `PhoneService`, `MultipleLines` |
| **Internet & Add-ons** | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Identifier** | `customerID` (Primary key; removed during cleaning to prevent spurious correlations) |

---

## 3. Full Project Architecture & Directory Layout

```text
telco-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Original raw dataset (7,043 records)
│   └── processed/
│       ├── cleaned_telco_churn.csv                # Cleaned dataset (7,043 rows, 20 columns)
│       └── engineered_telco_churn.csv             # Feature-engineered dataset (7,043 rows, 29 columns)
│
├── notebooks/
│   └── telco_churn_analysis.ipynb                 # Fully executed primary Jupyter Notebook (53 cells, rich outputs)
│
├── src/
│   ├── __init__.py                                # Python package initializer
│   ├── data_cleaning.py                           # TotalCharges blank handling, type casting, IQR audit
│   ├── feature_engineering.py                     # TenureGroup, TotalServices, contract flags, spend metrics
│   ├── preprocessing.py                           # ColumnTransformer, StandardScaler, OneHotEncoder, stratified split
│   ├── model_training.py                          # Pipeline assembly, Baseline vs Engineered experiments
│   └── evaluation.py                              # Confusion matrices, ROC/PR curves, odds ratios, plots
│
├── outputs/
│   ├── plots/                                     # Saved high-resolution figures
│   │   ├── confusion_matrix_baseline.png
│   │   ├── confusion_matrix_engineered.png
│   │   ├── confusion_matrix_engineered_balanced.png
│   │   ├── roc_curve_comparison.png
│   │   ├── precision_recall_curve_comparison.png
│   │   ├── model_metric_comparison_barchart.png
│   │   ├── actual_vs_predicted_engineered.png
│   │   └── top_coefficients_engineered.png
│   ├── metrics/                                   # Real model comparison tables & audit logs
│   │   ├── model_comparison_metrics.csv
│   │   ├── baseline_model_coefficients_and_odds.csv
│   │   ├── engineered_model_coefficients_and_odds.csv
│   │   └── data_quality_audit.json
│   └── models/                                    # Serialized Scikit-learn pipelines (.joblib)
│       ├── logistic_regression_baseline.joblib
│       ├── logistic_regression_engineered.joblib
│       └── logistic_regression_engineered_balanced.joblib
│
├── app.py                                         # Interactive Streamlit dashboard
├── run_pipeline.py                                # End-to-end automated validation runner
├── generate_notebook.py                           # Programmatic notebook generator script
├── requirements.txt                               # Specific project dependencies
├── README.md                                      # GitHub repository README
├── PROJECT_DOCUMENTATION.md                       # Comprehensive documentation & execution manual
└── .gitignore                                     # Git ignore rules for caches and checkpoints
```

---

## 4. Phase 1: Exploratory Data Analysis (EDA) Findings

The exploratory analysis performed in `notebooks/telco_churn_analysis.ipynb` yielded key behavioral insights:

### 1. Tenure vs. Churn
- **Insight:** Over 50% of all customer churn occurs within the first 12 months (`tenure` $\le 12$).
- **Business Rationale:** Early customer onboarding, initial service quality, and setup assistance dictate long-term retention. After 24 months, churn rates decline significantly.

### 2. Monthly Charges vs. Churn
- **Insight:** Retained customers have a median monthly bill of ~$65, whereas churned customers have a median monthly bill of ~$80.
- **Business Rationale:** Higher price points increase consumer price sensitivity. When monthly charges climb above $70/month without perceived service value, churn propensity escalates.

### 3. Contract Type vs. Churn
- **Insight:** 
  - **Month-to-month contracts:** **42.7%** churn rate.
  - **One-year contracts:** **11.3%** churn rate.
  - **Two-year contracts:** **2.8%** churn rate.
- **Business Rationale:** Contractual commitment is the single strongest protective barrier against customer attrition.

### 4. Payment Method vs. Churn
- **Insight:** Customers paying via **Electronic Check** churn at an alarming rate of **45.3%**, compared to under **18%** for automated bank transfers and credit cards.
- **Business Rationale:** Electronic check payments require manual monthly action and frequently encounter billing friction, whereas automated payments run passively in the background.

---

## 5. Phase 2: Data Cleaning & Quality Assurance

### 1. The `TotalCharges` Whitespace Problem
- **Detection:** In the raw dataset, `TotalCharges` was loaded as an `object` (string) column rather than `float64`. Inspecting non-numeric entries identified **11 records** containing pure whitespace (`" "`).
- **Domain Root Cause:** When filtering for these 11 rows:
  ```text
  customerID  tenure  MonthlyCharges  TotalCharges  Contract  Churn
  7562-GSUVA       0           56.15                Two year     No
  4472-LVYGI       0           52.55                Two year     No
  3115-CZMZD       0           20.25                Two year     No
  ... (11 rows total)
  ```
  Every single customer with blank `TotalCharges` has **`tenure == 0`**!
- **Academic Viva Rationale for Imputing 0.0:**
  1. *Why not drop rows?* Dropping valid customer rows introduces survivorship bias and eliminates newly onboarded customers.
  2. *Why not impute with median/mean (~$2,283)?* Imputing $2,283 onto day-zero subscribers creates false data: a customer who has been subscribed for 0 days has not yet incurred thousands of dollars of billing.
  3. *Why 0.0 is mathematically sound:* Setting `TotalCharges = 0.0` accurately reflects that cumulative billed revenue at month zero is $0.00.

### 2. Outlier Analysis using Interquartile Range (IQR)
The IQR method ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$) was applied to numerical features:
- `tenure`: $Q_1 = 9.0$, $Q_3 = 55.0$, $\text{IQR} = 46.0 \implies \text{Outliers} = 0$ (0.0%).
- `MonthlyCharges`: $Q_1 = 35.5$, $Q_3 = 89.85$, $\text{IQR} = 54.35 \implies \text{Outliers} = 0$ (0.0%).
- `TotalCharges`: $Q_1 = 401.45$, $Q_3 = 3794.74$, $\text{IQR} = 3393.29 \implies \text{Outliers} = 0$ (0.0%).
- **Conclusion:** Telecom charges span legitimate commercial ranges ($0 to $8,684.80). Retaining all observations prevents artificial distortion of natural variance.

### 3. Identifier Removal
`customerID` was dropped prior to training to ensure the model does not learn spurious associations from primary keys.

---

## 6. Phase 3: Domain-Specific Feature Engineering

To provide richer behavioral and contractual signals to the Logistic Regression model, 9 domain-specific features were created in `src/feature_engineering.py`:

| Feature Name | Type | Formula / Logic | Business Context |
| :--- | :--- | :--- | :--- |
| `TenureGroup` | Categorical | Binned tenure: `New customer` (0-12m), `Short-term` (13-24m), `Medium-term` (25-48m), `Long-term` (>48m) | Represents customer lifecycle stages and early attrition risk. |
| `TotalServices` | Integer (0–9) | Count of: Phone, Lines, Internet, Security, Backup, Device Protection, Tech Support, TV, Movies | Product engagement/stickiness; multi-product customers face higher switching costs. |
| `HasOnlineSecurity` | Binary (0/1) | $1$ if `OnlineSecurity == 'Yes'`, else $0$ | Core security anchor associated with lower churn. |
| `HasOnlineBackup` | Binary (0/1) | $1$ if `OnlineBackup == 'Yes'`, else $0$ | Cloud storage dependency indicator. |
| `HasTechSupport` | Binary (0/1) | $1$ if `TechSupport == 'Yes'`, else $0$ | Technical assistance reducing unresolved service frustration. |
| `HasMultipleServices`| Binary (0/1) | $1$ if `TotalServices > 1`, else $0$ | Multi-product cross-sell indicator. |
| `IsLongTermContract`| Binary (0/1) | $1$ if `Contract` in `['One year', 'Two year']`, else $0$ | Explicit commitment flag versus Month-to-month. |
| `IsAutomaticPayment`| Binary (0/1) | $1$ if `PaymentMethod` in `['Bank transfer (automatic)', 'Credit card (automatic)']`, else $0$ | Eliminates manual check friction and involuntary churn. |
| `AverageMonthlySpend`| Continuous ($) | $\frac{\text{TotalCharges}}{\text{tenure}}$ (if $\text{tenure} > 0$), else `MonthlyCharges` | Historical bill pace, safely handling zero division. |

---

## 7. Phase 4: Preprocessing & Leak-Free Architecture

### Preventing Data Leakage
Data leakage occurs when test set distribution parameters (such as mean, variance, or category frequencies) leak into the training process:
1. An **80:20 Stratified Split** (`random_state=42`) is executed first on the dataset.
2. A Scikit-learn `ColumnTransformer` is assembled:
   - **Numerical Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
   - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. The complete `Pipeline` calls `fit()` **strictly on `X_train`**.
4. `X_test` is strictly transformed using the parameters learned from `X_train`.

### Why Scaling is Vital for Logistic Regression
Logistic Regression optimizes a regularized objective function:
$$\min_{\beta} -\sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right] + \frac{\lambda}{2} \sum_{j=1}^p \beta_j^2$$
Because the penalty parameter $\lambda$ is applied uniformly across all feature weights $\beta_j$:
- If unscaled, a feature with large numerical values like `TotalCharges` (range: $0 to $8,684) would require a very small coefficient $\beta_j \approx 0.0001$ to keep predictions in reasonable logit bounds, rendering the regularization penalty on $\beta_j^2$ virtually zero.
- Conversely, binary 0/1 features would receive disproportionately large penalties.
- `StandardScaler` standardizes each feature to $\mu = 0, \sigma = 1$, ensuring **equal regularized shrinkage** and **stable solver convergence** (L-BFGS).

---

## 8. Phase 5: Logistic Regression Mathematical Formulation

Logistic Regression models the conditional probability $p(x) = P(Y=1 \mid X=x)$ using the logistic sigmoid link function:

$$p(x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \beta_0 + \sum_{j=1}^p \beta_j x_j$$

Inverting the sigmoid function yields the **logit** or **log-odds**:

$$\log\left(\frac{p(x)}{1 - p(x)}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

### Exponentiation to Odds Ratio (OR)
To interpret the coefficients in business terms, we exponentiate both sides:

$$\text{Odds} = \frac{p(x)}{1 - p(x)} = e^{\beta_0} \cdot e^{\beta_1 x_1} \cdot \dots \cdot e^{\beta_p x_p}$$

$$\text{Odds Ratio}_j = \frac{\text{Odds}(x_j + 1)}{\text{Odds}(x_j)} = e^{\beta_j}$$

- **$\beta_j > 0 \implies \text{OR}_j > 1$:** Every unit increase in $x_j$ multiplies the odds of customer churn by $\text{OR}_j$ (increases churn risk).
- **$\beta_j < 0 \implies \text{OR}_j < 1$:** Every unit increase in $x_j$ reduces the odds of customer churn by $(1 - \text{OR}_j) \times 100\%$ (retention anchor).
- **$\beta_j = 0 \implies \text{OR}_j = 1$:** The feature has no linear relationship with the log-odds of churn.

---

## 9. Phase 6: Model Evaluation & Experimental Comparison

Three experiments were executed on the **identical 1,409 unseen test instances**:

### Model Evaluation Results Table
| Model Specification | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Model** (Original Cleaned Features) | **80.55%** | **65.72%** | 55.88% | 0.6040 | 0.8420 |
| **Engineered Model** (Domain Engineered Features) | 79.91% | 65.32% | 51.87% | 0.5782 | **0.8423** |
| **Engineered (Balanced)** (`class_weight='balanced'`) | 73.31% | 49.83% | **79.14%** | **0.6116** | 0.8419 |

### Detailed Confusion Matrices (Test Set $N=1,409$)
```text
1. Baseline Model:
   Retained (0): 926 True Negatives  |  109 False Positives
   Churned  (1): 165 False Negatives |  209 True Positives

2. Engineered Model:
   Retained (0): 932 True Negatives  |  103 False Positives
   Churned  (1): 180 False Negatives |  194 True Positives

3. Engineered (Balanced Model):
   Retained (0): 737 True Negatives  |  298 False Positives
   Churned  (1):  78 False Negatives |  296 True Positives
```

### In-Depth Discussion: Why Accuracy is Misleading & Why Recall is Pivotal
1. **The 73.5% Majority Baseline:** In this dataset, 73.5% of customers do not churn. A naive "dummy" model that predicts "No Churn" for every single customer scores **73.46% accuracy** while intercepting **0% of churners** ($\text{Recall} = 0\%$).
2. **Asymmetric Business Costs:**
   - **Cost of a False Negative (Type II Error):** The model predicts a customer will stay, but they churn. The telecom company loses the entire Customer Lifetime Value ($1,000–$2,500+).
   - **Cost of a False Positive (Type I Error):** The model predicts a customer will churn, but they would stay. The telecom company reaches out with a check-in call, discount coupon, or loyalty reward ($5–$25 cost).
3. **The Power of Class Balancing:** Under `class_weight='balanced'`, Recall surges from **51.87% to 79.14%**, reducing False Negatives from **180 down to only 78** (capturing nearly 4 out of 5 churners in advance).

---

## 10. Phase 7: Model Interpretability & Odds Ratios

*(Conditioned associations — not causal relationships)*

### Top Features Increasing Churn Odds (Risk Drivers)
| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Practical Business Meaning |
| :--- | :---: | :---: | :--- |
| `InternetService_Fiber optic` | **+0.6531** | **1.9216** | Fiber Optic subscribers have **92.2% higher odds of churn** compared to baseline, indicating aggressive competitor promotions and price sensitivity around high monthly fiber bills. |
| `TenureGroup_Long-term` | **+0.3966** | **1.4868** | In models with separate continuous `tenure`, long-term categorical flags highlight late-lifecycle contract renewals. |
| `TotalServices` | **+0.3471** | **1.4150** | Subscribing to many unmanaged add-on services without proper setup can increase service frustration. |
| `TotalCharges` | **+0.2234** | **1.2503** | Cumulative spend accumulation correlates with customer fatigue over prolonged billing periods. |

### Top Features Decreasing Churn Odds (Retention Anchors)
| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Practical Business Meaning |
| :--- | :---: | :---: | :--- |
| `tenure` | **-1.1584** | **0.3140** | Every 1-standard-deviation increase in customer tenure decreases churn odds by **68.6%** ($1 - 0.3140$). |
| `InternetService_DSL` | **-0.6557** | **0.5191** | DSL subscribers have **48.1% lower odds of churn** than fiber optic subscribers (stable, budget-friendly segment). |
| `Contract_Two year` | **-0.5285** | **0.5895** | Two-year contractual commitments reduce churn odds by **41.1%**. |
| `IsLongTermContract` | **-0.4263** | **0.6529** | Any annual or multi-year contract reduces churn odds by **34.7%** compared to month-to-month contracts. |

---

## 11. Phase 8: Step-by-Step Instructions to Launch Notebook & Dashboard

### Prerequisites
Make sure Python 3.10+ is installed on your computer.

### Step 1: Open the Project Directory
Open PowerShell or your preferred terminal:
```powershell
cd c:\Predsem5\telco-churn-prediction
```

### Step 2: Install Required Dependencies
Run:
```powershell
pip install -r requirements.txt
```

---

### Step 3: Run the Automated Pipeline Script
To execute data cleaning, feature engineering, model training, metric calculation, and plot exporting in one step:
```powershell
python run_pipeline.py
```
**Expected Output:**
```text
============================================================
TELCO CHURN PREDICTION – END-TO-END PIPELINE EXECUTION
============================================================
[Step 1/7] Loading raw data...
[Step 2/7] Auditing and cleaning data...
Blank TotalCharges detected: 11
Cleaned dataset saved: (7043, 20)
[Step 3/7] Performing domain feature engineering...
Engineered dataset shape: (7043, 29)
[Step 4/7] Training Logistic Regression models across experiments...
Pipeline saved to: outputs\models\logistic_regression_baseline.joblib
Pipeline saved to: outputs\models\logistic_regression_engineered.joblib
Pipeline saved to: outputs\models\logistic_regression_engineered_balanced.joblib
[Step 5/7] Evaluating models on identical test partition...
Model Evaluation Comparison Table:
                Model  Accuracy  Precision  Recall  F1-score  ROC-AUC
       Baseline Model    0.8055     0.6572  0.5588    0.6040   0.8420
     Engineered Model    0.7991     0.6532  0.5187    0.5782   0.8423
Engineered (Balanced)    0.7331     0.4983  0.7914    0.6116   0.8419
[Step 6/7] Generating and saving evaluation charts...
[Step 7/7] Computing model coefficients and odds ratios...
Pipeline execution completed successfully!
```

---

### Step 4: Launch the Executed Jupyter Notebook
To view the fully executed, publication-ready Jupyter Notebook:
```powershell
jupyter notebook notebooks/telco_churn_analysis.ipynb
```
*(The notebook contains 53 executed cells, full Markdown documentation, and all rendered Seaborn/Matplotlib figures).*

---

### Step 5: Launch the Interactive Streamlit Dashboard
To run the interactive web application:
```powershell
streamlit run app.py
```
The browser will automatically open:
```text
Local URL: http://localhost:8501
```

#### How to Use the Streamlit Application:
1. **Executive Overview Tab:** View high-level metrics (Total Customers: 7,043, Retention: 73.5%, Churn: 26.5%) and key distribution charts.
2. **Data Quality & Cleaning Tab:** Inspect the 11 blank `TotalCharges` records and verify the IQR outlier report.
3. **Feature Engineering Tab:** Explore the catalog of the 9 engineered features and their distributions.
4. **Model Performance Tab:** View the comparative metric table, confusion matrices, ROC curves, PR curves, and Odds Ratio tables.
5. **Customer Churn Predictor Tab:**
   - Select either **Standard Model** or **Recall-Optimized Model**.
   - Adjust customer attributes using interactive sliders and dropdowns (e.g. Tenure: 6 months, Contract: Month-to-month, Internet: Fiber optic, Payment: Electronic check).
   - Click **⚡ Predict Churn Probability**.
   - The application passes the row directly into the serialized Scikit-learn `Pipeline` and displays:
     - Predicted Class Badge (`High Risk Churn` or `Likely Retained`).
     - Churn Probability & Retention Probability gauge bar.
     - Tailored retention action recommendations based on the customer's risk factors.

---

## 12. Academic Viva Defense Questions & Answers

### Q1: Why use Logistic Regression instead of Random Forest or XGBoost?
> **Answer:** Logistic Regression is the standard benchmark for binary classification because it outputs well-calibrated posterior probabilities and provides exact mathematical transparency. By examining coefficients and calculating Odds Ratios ($e^\beta$), we can explicitly explain to stakeholders which features increase or decrease the odds of churning. In customer retention, understanding *why* a customer is at risk is just as important as predicting the risk.

### Q2: How did you ensure there was no data leakage in your preprocessing?
> **Answer:** We performed an 80:20 stratified train-test split first. All feature scaling statistics (`StandardScaler` mean and standard deviation) and categorical mappings (`OneHotEncoder` categories) were fitted strictly on `X_train`. The test set `X_test` was only transformed. Furthermore, all engineered features were computed purely from existing predictor columns without utilizing the target column `Churn`.

### Q3: Why did you set TotalCharges to 0.0 for 11 records instead of dropping them or imputing with the median?
> **Answer:** An inspection of the data revealed that all 11 customers with blank `TotalCharges` had `tenure == 0`. These are newly registered subscribers who enrolled during the current billing period and have not yet completed a billing cycle. Setting `TotalCharges = 0.0` accurately reflects that cumulative billed revenue at month zero is $0.00. Imputing with the median (~$2,283) would corrupt customer spending history by assigning thousands of dollars of non-existent charges.

### Q4: Why does the balanced model have lower accuracy but higher recall?
> **Answer:** In imbalanced classification, accuracy is dominated by the majority class (73.5% non-churners). By applying `class_weight='balanced'`, the optimization objective penalizes False Negatives (missing a churner) more heavily than False Positives. This shifts the decision boundary to flag borderline cases, increasing Recall from **51.87% to 79.14%** while slightly reducing overall accuracy to 73.31%. For a telecom company, this trade-off is highly advantageous because losing a customer is far more costly than sending an unnecessary retention perk.

### Q5: Can Logistic Regression coefficients be interpreted as causal effects?
> **Answer:** No. Logistic regression coefficients represent statistical associations (conditional on all other variables in the model), not causal relationships. For example, Fiber Optic internet has a positive coefficient ($\text{OR} = 1.92$), meaning Fiber Optic users are associated with higher churn odds, but Fiber Optic service does not necessarily *cause* churn; the association is confounded by higher monthly charges and competitor fiber marketing.

---

## 13. Limitations & Future Enhancements

1. **Synthetic Minority Oversampling (SMOTE):** Future work can explore SMOTE or ADASYN to evaluate whether generating synthetic minority instances produces a better precision-recall curve than class weighting.
2. **Out-of-Time (OOT) Temporal Validation:** If timestamped cohort data becomes available, transitioning from random stratified splitting to temporal validation will better simulate production deployment over calendar quarters.
3. **Non-linear Ensemble Benchmarks:** While Logistic Regression is preferred for interpretability, comparing its performance with LightGBM or XGBoost combined with SHAP (SHapley Additive exPlanations) would quantify the performance gain of non-linear interaction terms.
4. **Survival Analysis:** Implementing Cox Proportional Hazards modeling to predict not just *if* a customer will churn, but *when* they are most likely to churn.

---

## 🔗 Quick Reference Links
- [Jupyter Notebook: telco_churn_analysis.ipynb](file:///c:/Predsem5/telco-churn-prediction/notebooks/telco_churn_analysis.ipynb)
- [Streamlit Dashboard: app.py](file:///c:/Predsem5/telco-churn-prediction/app.py)
- [Pipeline Runner: run_pipeline.py](file:///c:/Predsem5/telco-churn-prediction/run_pipeline.py)
- [GitHub Repository](https://github.com/gxcfr/TELCO_dataset)
