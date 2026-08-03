# Prediction Tool for Early Adverse Outcomes After Surgery in Early-Onset Rectal Cancer

This repository contains the R source code used in the study:

**A Machine Learning Model for Predicting Early Adverse Outcomes After Surgery in Patients with Early-Onset Rectal Cancer: A Two-Center Retrospective Cohort Study**

Submitted to *World Journal of Surgical Oncology* (Manuscript ID: 53552edc-8849-45f8-83e7-5a09f59f9115)

---

## 📌 Overview

This repository provides the complete R scripts used for:

- Data preprocessing and missing value imputation (MICE)
- Spearman correlation filtering and Boruta feature selection
- Training of 12 machine learning models (decision tree, random forest, XGBoost, LightGBM, SVM, MLP, logistic regression, KNN, LASSO, elastic net, ridge regression, and stacking ensemble)
- Bayesian hyperparameter optimization with 5-fold cross-validation
- Internal validation (stratified 7:3 split) and independent external validation
- SHAP (SHapley Additive exPlanations) analysis for global and local model interpretability
- Decision curve analysis (DCA)
- Calibration, discrimination, and confusion matrix evaluation
- Figure generation (model comparison heatmaps, calibration plots, SHAP dependence/waterfall plots, correlation matrices)
- Development of the Shiny web-based prediction calculator

The web application is available at:  
🔗 **[https://qddx-gi.shinyapps.io/EORC_EAO/](https://qddx-gi.shinyapps.io/EORC_EAO/)**

---

## 🔒 Data Availability

The clinical dataset used in this study contains sensitive patient information and is **not publicly available**.

De-identified data may be made available from the corresponding author upon reasonable request and institutional approval.

**Contact:**

- Prof. Yan-Bing Zhou  
  Email: qdfyzhouyanbing@163.com

> ⚠️ **Note for Reviewers:** To facilitate reproducibility assessment during the peer-review process, the code repository access can be provided to reviewers upon request prior to publication. Please contact the corresponding authors for access credentials.

---

## 📂 Repository Structure

├── R/ # All R scripts for analysis and model construction

│ ├── 01_data_preprocessing.R # Data cleaning, MICE imputation

│ ├── 02_feature_selection.R # Spearman correlation filtering, Boruta algorithm

│ ├── 03_model_training.R # Training of 12 ML models with Bayesian optimization

│ ├── 04_model_evaluation.R # Internal & external validation, calibration, DCA

│ ├── 05_shap_analysis.R # SHAP global importance & local waterfall plots

│ ├── 06_figures.R # Figure generation scripts

│ └── app.R # Shiny web application source code

└── README.md

---

## 🖥 Shiny Web Calculator

### Access

The interactive prediction tool is deployed at:  
[https://qddx-gi.shinyapps.io/EORC_EAO/](https://qddx-gi.shinyapps.io/EORC_EAO/)

### Required Input Variables

The calculator requires the following 11 predictor variables, all of which are routinely available in postoperative clinical practice once pathological examination is complete:

| Variable | Type | Description |
|----------|------|-------------|
| **pT** | Categorical | Pathological T stage (T0, T1, T2, T3, T4a, T4b) |
| **pTNM** | Categorical | Pathological TNM stage (I, II, III, IV) |
| **Lymphovascular invasion (LVI)** | Binary | Presence (Yes/No) of lymphovascular invasion on pathology |
| **Neoadjuvant therapy** | Binary | Whether the patient received neoadjuvant therapy prior to surgery (Yes/No) |
| **MMR status** | Categorical | Mismatch repair status: dMMR (deficient) or pMMR (proficient) |
| **Crea** | Continuous | Serum creatinine (mg/dL) |
| **CEA** | Continuous | Carcinoembryonic antigen (ng/mL) |
| **CA199** | Continuous | Carbohydrate antigen 19-9 (U/mL) |
| **BUN/Crea** | Continuous | Blood urea nitrogen-to-creatinine ratio |
| **Urea/Crea** | Continuous | Urea-to-creatinine ratio |
| **Tumor size** | Continuous | Maximum tumor diameter (cm) |

### How to Use

1. Open the web application using the link above
2. Enter the patient's postoperative pathological and laboratory values into the corresponding input fields on the left sidebar
3. Click the **"Predict"** button
4. The predicted probability of early adverse outcomes (EAO) within one year after surgery will be displayed on the right panel

### Interpreting Results

- **Output:** The calculator returns an individualized predicted probability (0–100%) of experiencing early adverse outcomes (EAO) within one year after surgery.
- **EAO definition:** A composite endpoint comprising any of the following within one year postoperatively:
  - Clavien-Dindo grade ≥III postoperative complications
  - Locoregional recurrence or distant metastasis
  - All-cause death
- **Interpretation guidance:** A higher predicted probability indicates a greater risk of EAO and may warrant:
  - Intensified postoperative surveillance (e.g., shorter follow-up intervals)
  - Multidisciplinary discussion regarding adjuvant therapy considerations
  - Individualized prognosis communication with the patient and family

> **Note:** The model is designed for **postoperative** use. It integrates pathological staging data available after surgical resection with clinical and laboratory variables and is not intended for preoperative risk prediction.

---

## 🖥 Software Environment

- **R** version 4.4.1
- **Core packages:**
  - `tidyverse` — data manipulation and visualization
  - `tidymodels` — unified modeling framework
  - `mice` — multiple imputation with chained equations
  - `Boruta` — wrapper-based feature selection
  - `pROC` — ROC curve analysis
  - `fastshap` / `DALEX` — SHAP value computation and model explainability
  - `ggplot2` — figure generation
  - `shiny` — web application framework
  - `BayesianOptimization` — hyperparameter tuning
- **Additional packages:** `xgboost`, `lightgbm`, `randomForest`, `glmnet`, `kernlab`, `nnet`, `caret`, `rms`, `rmda`, `gridExtra`, `corrplot`

Full session information is provided in `sessionInfo.txt`.

---

## 📖 Citation

If you use this code, please cite:

Yue XT, Zhang GJ, Cao H, Zhao XY, Sun YQ, Zhang XQ, Zhou YB. A Machine Learning Model for Predicting Early Adverse Outcomes After Surgery in Patients with Early-Onset Rectal Cancer: A Two-Center Retrospective Cohort Study. World Journal of Surgical Oncology (under review).
---

## 🏷 Abbreviations

- **EORC:** Early-onset rectal cancer
- **EAO:** Early adverse outcomes
- **SHAP:** SHapley Additive exPlanations
- **AUC:** Area under the receiver operating characteristic curve
- **DCA:** Decision curve analysis
- **MICE:** Multiple imputation with chained equations
- **LVI:** Lymphovascular invasion
- **MMR:** Mismatch repair
- **pT:** Pathological T stage
- **pTNM:** Pathological TNM stage
