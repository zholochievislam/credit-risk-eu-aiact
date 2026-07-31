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
│   ├── modeling.py        # LightGBM training, OOF predictions & model persistence (joblib)
│   ├── explainability.py  # SHAP log-odds conversion & text report generation
│   ├── fairness.py        # Fairlearn audit pipeline, DIR/EOD metrics & per-group thresholds
│   └── oversight.py       # Human Oversight Gate (Article 14) & decision audit logging (Article 12)
│
├── models/                # Persisted model artifacts (final_model.joblib, thresholds, metrics)
├── logs/                  # Decision audit trail (decisions_log.jsonl)
├── tests/                 # pytest suite (data pipeline, leakage checks, applicant routing)
├── app.py                 # Streamlit web application (Day 6)
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
 
### 5. Human Oversight Gate & Audit Trail *(Day 6)*
 
**Challenge:** The EU AI Act (Article 14) requires high-risk automated decisions to include meaningful human oversight rather than fully automated approval/rejection for every case. It also requires a persistent, reviewable record of every decision made (Article 12).
 
**Solution:** Built `src/oversight.py` as a three-way decision router:
* For each applicant, the model's predicted default probability is compared against **that applicant's group-specific threshold** (from the Day 5 mitigation) with an uncertainty margin (±0.10) around it.
* **Below the lower bound** → `AUTO_APPROVE`. **Above the upper bound** → `AUTO_REJECT`. **Within the margin** (the model is not confident) → `MANUAL_REVIEW`, routed to a human loan officer.
* Every decision — inputs, computed risk, threshold used, and outcome — is appended to a JSONL audit log (`logs/decisions_log.jsonl`), creating a persistent, timestamped record for compliance review.
**Validation:** Tested against multiple synthetic applicant profiles (low-risk, high-risk, borderline, and an extreme "very safe" profile). One interesting finding: two structurally different but both extremely low-risk applicants produced *identical* risk scores to 6 decimal places. Diagnosed empirically (confirming the preprocessed feature vectors were genuinely different, not a bug) — this is expected behavior for gradient-boosted trees, which partition risk space into discrete regions rather than a continuous function; profiles landing in the same "leaf" across all 100 trees receive identical scores. Documented as a known model characteristic rather than treated as a defect.
 
### 6. Interactive Web Application (Streamlit) *(Day 6)*
 
Built `app.py` as a full-stack interface unifying every backend module into a single decision-support tool for loan officers and compliance auditors:
 
* **Sidebar — Applicant Information:** Input form for all applicant attributes. The Loan-to-Income Ratio is *computed automatically* from income and loan amount (rather than entered manually) to eliminate the risk of internally inconsistent inputs being fed to the model.
* **Tab 1 — Scoring Result:** Displays the computed default probability and the routed decision (`AUTO_APPROVE` / `AUTO_REJECT` / `MANUAL_REVIEW`), color-coded for at-a-glance reading.
* **Tab 2 — Explainability (Article 13):** Live SHAP waterfall plot and a plain-language explanation of the top risk drivers for the specific applicant just evaluated — the same Adverse Action Notice logic built on Day 4, now served on-demand for any new applicant.
* **Tab 3 — Compliance & Governance Dashboard:** Built for auditors rather than loan officers. Shows the full decision audit trail (read live from the JSONL log) plus a summary of the Day 5 fairness audit — Disparate Impact Ratio and Equal Opportunity Difference for `person_home_ownership`, shown **before vs. after** threshold mitigation, side by side.
---
 
## 🚀 Upcoming Steps (Days 7 – 9)
 
- [ ] **Day 7:** Polish the Streamlit UI (visual styling, layout refinement); expand the Compliance dashboard with additional fairness visualizations.
- [ ] **Day 8:** Formal ML **Model Card** — model purpose, training data, performance metrics, fairness audit results, known limitations (data leakage findings, dataset synthetic nature, tree-saturation behavior), and monitoring recommendations. Finalize `requirements.txt` and run the full `pytest` suite.
- [ ] **Day 9:** Code freeze, final README pass with application screenshots, repository cleanup, and CV/networking follow-up.
---
 
## 🚀 How to Run the Pipeline Locally
 
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```
 
2. Set up a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```
 
3. Run the notebooks in order (to reproduce the trained model, thresholds, and metrics artifacts under `models/`):
```bash
jupyter notebook notebooks/03_explainability.ipynb
jupyter notebook notebooks/04_fairness.ipynb
```
 
4. Launch the Streamlit application:
```bash
streamlit run app.py
```
This opens the app locally at `http://localhost:8501`. *(A public deployment link will be added here once the app is deployed to Streamlit Community Cloud.)*
 
---
 
*Developed as part of an Advanced End-to-End Machine Learning Engineering Portfolio.*
