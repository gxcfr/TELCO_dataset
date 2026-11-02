# Telco Customer Churn Prediction: End-to-End Machine Learning Pipeline

An academically presentable, reproducible, and production-grade Machine Learning project analyzing and predicting customer attrition using the **IBM Telco Customer Churn** dataset and **Logistic Regression** modeling.

---

## 📋 Table of Contents
1. [Project Overview & Business Problem](#1-project-overview--business-problem)
2. [Dataset Description](#2-dataset-description)
3. [Project Architecture & Repository Structure](#3-project-architecture--repository-structure)
4. [Data Cleaning & Quality Assurance](#4-data-cleaning--quality-assurance)
5. [Domain Feature Engineering](#5-domain-feature-engineering)
6. [Preprocessing & Leak-Free Pipeline](#6-preprocessing--leak-free-pipeline)
7. [Logistic Regression Mathematical Formulation](#7-logistic-regression-mathematical-formulation)
8. [Experimental Design & Model Evaluation](#8-experimental-design--model-evaluation)
9. [Model Interpretability & Odds Ratios](#9-model-interpretability--odds-ratios)
10. [Interactive Streamlit Dashboard](#10-interactive-streamlit-dashboard)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [Academic Viva Defense Q&A](#12-academic-viva-defense-qa)
13. [Limitations & Future Improvements](#13-limitations--future-improvements)

---

## 1. Project Overview & Business Problem

In the telecommunications sector, acquiring new customers costs between **5 to 7 times more** than retaining existing ones. Customer churn directly impacts:
- Monthly Recurring Revenue (MRR)
- Customer Lifetime Value (CLV)
- Overall enterprise valuation

The primary objective of this project is to implement the **complete Data Science lifecycle**:
- **Exploratory Data Analysis (EDA):** Detecting behavioral patterns and customer churn risk indicators.
- **Data Cleaning:** Resolving whitespace formatting anomalies in `TotalCharges` through sound domain logic.
- **Domain Feature Engineering:** Building engagement, lifecycle, and contract stickiness predictors.
- **Leak-Free Preprocessing:** Standard scaling and one-hot encoding fitted strictly on training data.
- **Controlled Modeling:** Comparing Baseline vs. Feature-Engineered vs. Balanced Logistic Regression models.
- **Evaluation & Interpretation:** Emphasizing why Recall is the pivotal metric in churn prediction and interpreting feature coefficients via **Odds Ratios** ($OR = e^\beta$).

---

## 2. Dataset Description

- **Source:** [IBM Telco Customer Churn Dataset (Kaggle / IBM Sample Data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Expected Filename:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Total Records:** 7,043 customers
- **Total Features:** 21 columns (20 predictors + 1 target)
- **Target Variable:** `Churn`
  - `No` $\rightarrow 0$ (73.46% of dataset — Retained customers)
  - `Yes` $\rightarrow 1$ (26.54% of dataset — Churned customers)

### Key Feature Categories:
1. **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
2. **Account / Contract:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
3. **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
4. **Identifier:** `customerID` (Removed during data cleaning to prevent spurious correlations)

---

## 3. Project Architecture & Repository Structure

```text
telco-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Raw dataset (7,043 rows)
│   └── processed/
│       ├── cleaned_telco_churn.csv                # Cleaned dataset (7,043 rows, 20 cols)
│       └── engineered_telco_churn.csv             # Feature-engineered dataset (7,043 rows, 29 cols)
│
├── notebooks/
│   └── telco_churn_analysis.ipynb                 # Comprehensive, fully-executed Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                           # TotalCharges handling, type casting, outlier analysis
│   ├── feature_engineering.py                     # TenureGroup, TotalServices, contract & spend features
│   ├── preprocessing.py                           # ColumnTransformer, StandardScaler, OneHotEncoder, split
│   ├── model_training.py                          # Pipeline assembly, Baseline vs Engineered experiments
│   └── evaluation.py                              # Confusion matrices, ROC, PR curves, odds ratios
│
├── outputs/
│   ├── plots/                                     # Publication-quality charts (ROC, PR, CM, Coefficients)
│   │   ├── confusion_matrix_baseline.png
│   │   ├── confusion_matrix_engineered.png
│   │   ├── confusion_matrix_engineered_balanced.png
│   │   ├── roc_curve_comparison.png
│   │   ├── precision_recall_curve_comparison.png
│   │   ├── model_metric_comparison_barchart.png
│   │   ├── actual_vs_predicted_engineered.png
│   │   └── top_coefficients_engineered.png
│   ├── metrics/                                   # Evaluation metrics CSVs and JSON audits
│   │   ├── model_comparison_metrics.csv
│   │   ├── baseline_model_coefficients_and_odds.csv
│   │   ├── engineered_model_coefficients_and_odds.csv
│   │   └── data_quality_audit.json
│   └── models/                                    # Serialized Scikit-learn pipelines
│       ├── logistic_regression_baseline.joblib
│       ├── logistic_regression_engineered.joblib
│       └── logistic_regression_engineered_balanced.joblib
│
├── app.py                                         # Interactive Streamlit dashboard
├── run_pipeline.py                                # End-to-end automated validation runner
├── generate_notebook.py                           # Programmatic notebook generator script
├── requirements.txt                               # Explicit project dependencies
├── README.md                                      # Project documentation & viva guide
└── .gitignore                                     # Python/Jupyter cache ignore
```

---

## 4. Data Cleaning & Quality Assurance

### The `TotalCharges` Problem: Detection & Solution
1. **Detection:** When auditing data types, `TotalCharges` was typed as `object` (string). Checking for whitespace `r"^\s*$"` revealed **11 rows** containing blank space characters (`" "`).
2. **Domain Root Cause:** Cross-referencing these 11 customers revealed that **every single one had `tenure == 0`**. These are brand new customers who joined within the current billing cycle and have not yet completed a full month.
3. **Cleaning Strategy:**
   - **Incorrect approach:** Removing these rows introduces sample bias; imputing with mean or median (~$2,283) assigns thousands of dollars of phantom charges to day-zero subscribers.
   - **Chosen approach:** Replace whitespace with `0.0` and cast to `float64`. This accurately mirrors the cumulative billed charges at $t=0$.

### Outlier Inspection (IQR)
- Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) were analyzed using the Interquartile Range ($1.5 \times \text{IQR}$).
- **Result:** $0$ extreme anomalies in `tenure` and `MonthlyCharges`. `TotalCharges` exhibited a natural commercial right-skew. All rows were retained to avoid artificial variance reduction.

---

## 5. Domain Feature Engineering

Nine new predictive features were engineered without target leakage:

| Feature Name | Type | Mathematical / Logical Definition | Business Significance |
| :--- | :--- | :--- | :--- |
| `TenureGroup` | Categorical | Binned tenure: `New customer` (0-12m), `Short-term` (13-24m), `Medium-term` (25-48m), `Long-term` (>48m) | Segments customer lifecycle cohorts; churn hazard is highest in the first 12 months. |
| `TotalServices` | Integer (0–9) | Count of subscribed products: Phone, Lines, Internet, Security, Backup, DeviceProtection, TechSupport, TV, Movies | Product engagement/stickiness; customers with multiple services have higher switching barriers. |
| `HasOnlineSecurity` | Binary (0/1) | $1$ if `OnlineSecurity == 'Yes'`, else $0$ | Core security anchor associated with lower churn. |
| `HasOnlineBackup` | Binary (0/1) | $1$ if `OnlineBackup == 'Yes'`, else $0$ | Cloud storage dependency indicator. |
| `HasTechSupport` | Binary (0/1) | $1$ if `TechSupport == 'Yes'`, else $0$ | Responsive technical assistance reducing frustration-driven churn. |
| `HasMultipleServices`| Binary (0/1) | $1$ if `TotalServices > 1`, else $0$ | Multi-product cross-sell indicator. |
| `IsLongTermContract`| Binary (0/1) | $1$ if `Contract` in `['One year', 'Two year']`, else $0$ | Contractual commitment barrier vs. Month-to-month. |
| `IsAutomaticPayment`| Binary (0/1) | $1$ if `PaymentMethod` in `['Bank transfer (automatic)', 'Credit card (automatic)']`, else $0$ | Eliminates manual payment friction and involuntary churn. |
| `AverageMonthlySpend`| Continuous ($) | $\text{TotalCharges} / \text{tenure}$ (if $\text{tenure} > 0$), else $\text{MonthlyCharges}$ | Historical monthly expenditure pace, safely handling zero division. |

---

## 6. Preprocessing & Leak-Free Pipeline

### Preventing Data Leakage
Data leakage occurs when information outside the training dataset is used to create or tune the model.
- An **80:20 Stratified Split** (`random_state=42`) is executed first.
- `ColumnTransformer` fits scalers and encoders **strictly on `X_train`**.
- `X_test` is only passed through `transform()`.

### Why Scaling is Vital for Logistic Regression
Logistic Regression solves for weights $\beta$ under $L_2$ regularization:
$$\min_{\beta} -\sum_{i=1}^N \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right] + \frac{\lambda}{2} \|\beta\|_2^2$$
Because the penalty $\frac{\lambda}{2} \sum \beta_j^2$ is applied uniformly across all coefficients, unscaled variables with large numerical magnitudes (such as `TotalCharges` spanning $0–$8,684) would force their coefficients to be artificially small to avoid heavy penalties. Conversely, binary 0/1 features would dominate. `StandardScaler` standardizes all features to zero mean and unit variance ($\mu=0, \sigma=1$), guaranteeing uniform regularization and rapid gradient descent convergence.

---

## 7. Logistic Regression Mathematical Formulation

Logistic Regression models the probability $p = P(Y=1 \mid X)$ via the sigmoid (logistic) link function:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \beta_0 + \sum_{j=1}^p \beta_j X_j$$

Inverting the link function gives the **log-odds (logit)**:
$$\log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$$

Exponentiating yields the **Odds Ratio (OR)**:
$$\text{OR}_j = e^{\beta_j}$$
- If $\beta_j > 0 \implies \text{OR}_j > 1$: Every 1-unit increase in $X_j$ multiplies the odds of churning by $\text{OR}_j$.
- If $\beta_j < 0 \implies \text{OR}_j < 1$: Feature $X_j$ is a protective retention anchor.

---

## 8. Experimental Design & Model Evaluation

### Experimental Setup
Both experiments were tested on the **exact same 1,409 unseen customer test instances** (80:20 stratified split):
- **Experiment A (Baseline):** 20 original cleaned features.
- **Experiment B (Engineered):** 29 features (cleaned + domain engineered).
- **Experiment C (Engineered Balanced):** 29 features with `class_weight='balanced'` to evaluate cost-sensitive classification.

### Actual Model Results Table:
| Model Name | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Model** | **0.8055** | **0.6572** | 0.5588 | 0.6040 | 0.8420 |
| **Engineered Model** | 0.7991 | 0.6532 | 0.5187 | 0.5782 | **0.8423** |
| **Engineered (Balanced)** | 0.7331 | 0.4983 | **0.7914** | **0.6116** | 0.8419 |

### Why Accuracy is Misleading in Churn Prediction:
1. **The Majority Class Baseline:** With 73.5% non-churners, a dummy classifier predicting "No Churn" for every customer achieves **73.5% accuracy** while capturing **0% of churners**.
2. **Cost Asymmetry:**
   - **False Negative (Costly):** The model predicts the customer will stay ($0$), but they churn ($1$). The telecom provider loses recurring revenue ($500–$2,000+ CLV).
   - **False Positive (Benign):** The model predicts churn ($1$), but the customer would have stayed ($0$). The company provides a proactive check-in or retention discount, costing negligible resources.
3. **Recall Optimization:** Under `class_weight='balanced'`, the model intercepts **79.14% of all churning customers** (nearly 4 out of 5), representing a massive business win over default classification thresholds.

---

## 9. Model Interpretability & Odds Ratios

*(Conditioned associations — not causal relationships)*

### Top Predictors Increasing Churn Odds:
| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Interpretation |
| :--- | :---: | :---: | :--- |
| `InternetService_Fiber optic` | **+0.6531** | **1.9216** | Customers with Fiber Optic have **92.2% higher odds of churn** than baseline, reflecting competitive market alternatives and service pricing friction. |
| `TotalServices` | **+0.3471** | **1.4150** | Higher service complexity without proper technical onboarding correlates with elevated churn odds. |
| `TotalCharges` | **+0.2234** | **1.2503** | Cumulative financial commitment creates billing fatigue over extended lifespans. |

### Top Retention Anchors (Decreasing Churn Odds):
| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Interpretation |
| :--- | :---: | :---: | :--- |
| `tenure` | **-1.1584** | **0.3140** | Every standard deviation increase in tenure reduces churn odds by **68.6%**. |
| `InternetService_DSL` | **-0.6557** | **0.5191** | DSL subscribers have **48.1% lower odds of churn** (stable, low-cost utility segment). |
| `Contract_Two year` | **-0.5285** | **0.5895** | Two-year contractual commitment decreases churn odds by **41.1%**. |
| `IsLongTermContract` | **-0.4263** | **0.6529** | Annual or multi-year contracts provide a **34.7% reduction** in churn odds. |

---

## 10. Interactive Streamlit Dashboard

The project includes an interactive web application built with Streamlit (`app.py`), styled with a clean white/light theme and blue accents.

### Dashboard Modules:
1. **Executive Overview:** High-level KPIs (Total Customers, Retained, Churned %, Feature count) and interactive tenure/contract visualizations.
2. **Data Quality & Cleaning:** Transparency report detailing missing values, the 11 `TotalCharges` rows, and IQR bounds.
3. **Feature Engineering:** Catalog of all 9 engineered features with before/after comparisons and count distributions.
4. **Model Performance & Evaluation:** Side-by-side comparison tables, ROC curves, PR curves, and Odds Ratio tables.
5. **Interactive Customer Churn Predictor:** Enables entering demographic, contract, and service attributes to generate live inference using the **exact serialized Scikit-learn Pipeline** (`.predict_proba()`), with actionable risk mitigation suggestions.

---

## 11. How to Run the Project

### Prerequisites
- Python 3.10+ installed.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Machine Learning Pipeline
This validates the modules, cleans data, engineers features, trains models, computes metrics, and exports all figures:
```bash
python run_pipeline.py
```

### Step 3: Launch the Jupyter Notebook
Open the primary deliverable notebook:
```bash
jupyter notebook notebooks/telco_churn_analysis.ipynb
```
*(The notebook is already fully executed with rich plots and output tables).*

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 12. Academic Viva Defense Q&A

**Q1: Why did you choose Logistic Regression instead of Random Forest or XGBoost?**
> *Answer:* Logistic Regression serves as the gold-standard linear baseline for binary classification. It produces well-calibrated posterior probabilities and provides exact mathematical transparency through coefficients and **Odds Ratios** ($e^\beta$). In regulated enterprise and telecom environments, explaining *why* a customer is flagged as high-risk is as crucial as predictive accuracy.

**Q2: How did you ensure there was no data leakage in your preprocessing?**
> *Answer:* We strictly separated our dataset using an 80:20 stratified train-test split *before* computing scaling statistics or one-hot categories. We used a Scikit-Learn `ColumnTransformer` inside a `Pipeline`, ensuring `fit()` was executed strictly on `X_train`, and `transform()` was applied independently to `X_test`.

**Q3: Why did you set TotalCharges to 0.0 for 11 records instead of dropping them or imputing with the median?**
> *Answer:* An inspection of the data revealed that all 11 customers with blank `TotalCharges` had `tenure == 0`. These are newly registered subscribers who have not completed their first billing cycle. Setting `TotalCharges = 0.0` accurately reflects their accumulated billed revenue at time zero. Imputing with the median (~$2,283) would distort customer spending history.

**Q4: Why does the balanced model have a lower accuracy but a higher recall?**
> *Answer:* By assigning higher weights to minority positive instances (`class_weight='balanced'`), the optimization objective penalizes False Negatives more heavily. The decision boundary shifts to classify borderline cases as positive, increasing False Positives slightly (lowering precision and overall accuracy) but successfully capturing **79.1% of actual churners** compared to only 51.9%.

---

## 13. Limitations & Future Improvements

1. **Synthetic Data Augmentation (SMOTE):** Evaluating whether synthetic minority oversampling provides a superior precision-recall trade-off compared to class weighting.
2. **Probability Calibration:** Testing Platt Scaling or Isotonic Regression for customer segments with extreme feature combinations.
3. **Temporal Cross-Validation:** If timestamped cohort data becomes available, transitioning from random stratified splitting to out-of-time (OOT) validation.
4. **Ensemble Benchmarking:** Benchmarking the regularized Logistic Regression against non-linear models (e.g., LightGBM, CatBoost) using SHAP (SHapley Additive exPlanations).
