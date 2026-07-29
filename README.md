# End-to-End Fair & Explainable Credit Risk Assessment System
> **Compliance-Ready Machine Learning Pipeline for Automated Credit Decisioning**
 
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=flat)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=flat)
![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn-red?style=flat)
![EU AI Act](https://img.shields.io/badge/Compliance-EU_AI_Act-003399?style=flat)
 
---
 
## 📌 Executive Summary
 
Automated credit scoring systems frequently suffer from two critical flaws: **opacity ("black-box" decisions)** and **unintended algorithmic bias**. Under modern European regulatory frameworks (such as Article 13 and Article 9 of the **EU AI Act**), financial institutions are legally required to provide transparent justifications for credit denials and to actively audit and mitigate discriminatory outcomes across demographic groups.
 
This project delivers a **production-ready Credit Scoring Pipeline** that integrates high predictive power with strict Explainable AI (XAI) and fairness controls. It automates credit decisioning, generates legal-grade **Adverse Action Notices**, conducts multi-attribute bias audits, and applies **group-specific decision thresholds** to reduce disparate outcomes — before serving predictions via an interactive web dashboard.
 
---
 
## 🛠️ Tech Stack & Modular Architecture
 
* **Core & Data Wrangling:** `Python 3.10+`, `Pandas`, `NumPy`, `Scikit-Learn`
* **Predictive Engine:** `LightGBM` (Gradient Boosting)
* **Explainable AI (XAI):** `SHAP` (TreeExplainer, Waterfall & Dependence plots)
* **Responsible AI:** `Microsoft Fairlearn` (MetricFrame, Disparate Impact Ratio, Equal Opportunity Difference)
* **UI & Deployment:** `Streamlit` (Interactive MVP)
* **Version Control:** `Git` / `GitHub`
### 📂 Repository Structure
 
```text
credit-risk-project/
│
├── data/                  # Raw and processed credit datasets (32,000+ records)
├── notebooks/             # Step-by-step exploratory and validation notebooks
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_explainability.ipynb
│   └── 04_fairness.ipynb      # Fairness audit + group-threshold mitigation (Day 5)
│
├── src/                   # Production-grade backend modules
│   ├── data_processing.py # Automated cleaning, handling missing data & OHE
│   ├── modeling.py        # LightGBM training, OOF predictions & threshold optimization
│   ├── explainability.py  # SHAP log-odds conversion & text report generation
│   └── fairness.py        # Fairlearn audit pipeline, DIR/EOD metrics & per-group thresholds
│
├── app.py                 # Streamlit web application interface (Day 6)
└── README.md              # Project documentation
```
 
---
 
## 📊 Engineering Milestones & Key Results
 
### 1. Data Pipeline & Preventing Data Leakage *(Day 2)*
* **Challenge:** Raw financial data contained missing values, unrealistic anomalies (e.g. `person_age > 100`), and unformatted categorical variables.
* **Solution:** Built a modular preprocessing pipeline (`src/data_processing.py`) using `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` inside a `ColumnTransformer`, validated with `pytest`.
### 2. Predictive Modeling & Leakage Detection *(Day 3)*
* **Challenge:** An initial baseline model showed a suspiciously high Gini (0.88) — a red flag for a real-world credit dataset. Root-cause analysis identified `loan_grade` and `loan_int_rate` as **leakage features** (they encode the bank's own prior risk assessment).
* **Solution:** Dropped both features and re-validated using honest **Out-of-Fold (OOF)** predictions from 5-fold Stratified Cross-Validation — ensuring every prediction comes from a model that never saw that applicant during training.
* **Result:** **OOF-based Gini: 0.7336** — a realistic, defensible performance level for a compliant credit risk model.
### 3. Explainable AI (XAI) & EU AI Act Compliance *(Day 4)*
* **Challenge:** Raw model outputs (log-odds) are not interpretable, and the law mandates clear justifications for adverse credit decisions.
* **Solution:** Integrated **SHAP TreeExplainer** (`src/explainability.py`):
  * Converted raw log-odds into business-facing default probabilities via the sigmoid function.
  * Extracted the **Top-4 individual risk drivers** per applicant, with direction (increased/decreased) and strength (strongly/moderately/slightly) of impact.
  * Automatically generated **Adverse Action Notices** combining Waterfall plots with structured, human-readable explanations.
### 4. Algorithmic Fairness Audit & Bias Mitigation *(Day 5)*
 
**Challenge:** Post-training audits can reveal that a model, despite strong overall accuracy, treats demographic subgroups unequally — either in approval rates (**Disparate Impact**) or in its ability to correctly detect real defaulters (**Equal Opportunity**).
 
**Methodology:** Using OOF predictions, a single global decision threshold (0.220, derived via Youden's J statistic) was evaluated across three attributes: `age_group`, `person_home_ownership`, and `loan_intent`.
 
| Attribute | Disparate Impact Ratio (DIR) | Equal Opportunity Diff (EOD) | Status |
|---|---|---|---|
| **Age Group** (18-25 / 26-40 / 40+) | 0.96 | 0.04 | ✅ PASS |
| **Loan Intent** | 0.79 | 0.07 | ⚠️ Borderline (DIR) |
| **Home Ownership** | 0.68 | 0.40 | ⚠️ **FLAGGED** |
 
* **Age** showed no meaningful disparity — despite the theoretical proxy-bias risk identified during EDA (via `cb_person_cred_hist_length`), the model does not translate this into unequal outcomes.
* **Loan Intent** sits just under the 0.80 DIR threshold; given it is not a legally protected attribute, this is logged as an observation for ongoing monitoring rather than an immediate mitigation target.
* **Home Ownership** showed the most significant issue: applicants with a **mortgage** had their real defaults correctly identified only **38.3%** of the time (True Positive Rate), compared to **~79%** for renters and homeowners — a 40-point gap in prediction reliability between groups.
**Mitigation Applied:** Rather than using a single global threshold, a **per-group decision threshold** was computed (via Youden's J statistic, applied independently within each `person_home_ownership` segment):
 
| Group | Optimized Threshold |
|---|---|
| OWN | 0.159 |
| MORTGAGE | 0.163 |
| OTHER | 0.197 |
| RENT | 0.240 |
 
**Result — Before vs. After Mitigation (`person_home_ownership`):**
 
| Metric | Before | After | Change |
|---|---|---|---|
| Disparate Impact Ratio (DIR) | 0.68 | 0.74 | ⬆️ Improved |
| Equal Opportunity Diff (EOD) | 0.40 | 0.21 | ⬆️ Improved (~48% reduction) |
| Mortgage — True Positive Rate | 38.3% | 63.7% | ⬆️ Nearly doubled |
| Mortgage — Approval Rate | 88.0% | 68.0% | ⬇️ Stricter screening |
 
**Honest Trade-off Disclosure:** Lowering the mortgage-specific threshold substantially improved the model's ability to correctly flag real defaulters in this group, at the cost of a lower approval rate for mortgage holders overall (some previously-approved low-risk applicants are now declined). DIR did not fully clear the 0.80 benchmark, as the `OWN` group retains a notably higher approval rate (90.2%) relative to the rest. This residual gap is documented here rather than masked, consistent with the EU AI Act's emphasis on **transparent, ongoing risk management** (Article 9) over one-time metric optimization.
 
---
 
## 🖥 Web Application (Streamlit MVP)
*(Coming in Day 6)*
 
* An interactive dashboard allowing loan officers to input applicant details, instantly view default risk probabilities (via the group-calibrated thresholds above), and review SHAP-based explanations for every decision.
* Applicants falling into a mid-risk band will be automatically routed to a **Human Oversight Gate** (Article 14) for manual review rather than fully automated approval/rejection.
---
 
## 🚀 Upcoming Steps (Days 6 – 7)
 
- [ ] **Day 6 (Streamlit Dashboard):** Integration of backend modules into a full-stack user interface, applying group-specific thresholds and SHAP explanations to live applicant input.
- [ ] **Day 7 (Model Card & Final Packaging):** Finalizing the formal ML Model Card, performance documentation, and deployment guidelines.
---
 
## 🚀 How to Run the Pipeline Locally
 
1. Clone the repository:
```bash
git clone https://github.com/zholochievislam/credit-risk-eu-aiact.git
cd credit-risk-eu-aiact
```
 
2. Set up a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```
 
3. Run the notebooks in order:
```bash
jupyter notebook notebooks/03_explainability.ipynb
jupyter notebook notebooks/04_fairness.ipynb
```
 
---