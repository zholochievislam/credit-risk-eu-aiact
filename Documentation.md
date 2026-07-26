# 📄 Budapest Bootcamp 2026 by Islam Zholochiev

**Document Type:** Intensive Bootcamp Roadmap & Technical Specification  
**Author:** Islam Zholochiev (BSc Data Science & Economics Student)  
**Location:** Budapest, Hungary  
**Duration:** 9 Days (July 25 – August 2, 2026)  

> **Disclaimer:** This is an educational portfolio project simulating EU AI Act compliance workflows for credit scoring. It is not a certified, legally compliant, or production-deployed system.

---

## 📌 1. Why This Project?

With the **EU AI Act enforcement milestone taking effect in August 2026**, AI systems used in credit scoring and lending are officially classified as High-Risk AI Systems. Financial institutions across Europe are currently scrambling to build processes for model explainability, fairness auditing, and human oversight. 

As a Data Science and Economics student, I designed this 9-day solo bootcamp in Budapest to build a **compliance-informed credit scoring framework**. Instead of treating ML as a black box, this project demonstrates how technical engineering, ethical fairness, and European regulatory standards intersect in practice.

---

## 🎯 2. What I’m Building and Why

The goal is to build a transparent, fair, and explainable credit scoring pipeline that addresses key requirements of the EU AI Act (Articles 9, 10, 12, 13, and 14). 

Rather than chasing marginal accuracy gains, this project prioritizes **model interpretability (SHAP)**, **demographic fairness auditing (Fairlearn)**, and **human-in-the-loop decision routing**.

### Scope Management: MVP vs. Stretch vs. Cut List
To ensure a fully finished, high-quality deliverable within 9 days, project scope is strictly managed as follows:

* **MVP (Must Ship):**
  * Data pipeline with unit tests (`pytest`).
  * Baseline model (`LightGBM`) with cross-validation.
  * Explainable AI layer (`SHAP` global summary & local waterfall plots).
  * Pre-training data bias audit (Article 10) and post-training fairness mitigation (Article 9) on 1 sensitive attribute (`Age`) using 1 primary metric (Demographic Parity).
  * Bias mitigation using 1 primary method (`ThresholdOptimizer`).
  * Human-Oversight decision gate (routing uncertain predictions to manual review).
  * Clean, well-documented `README.md` with trade-off analysis.
* **Stretch Goals (Only if ahead of schedule by Day 7):**
  * Second model comparison (`LogisticRegression` baseline vs. `LightGBM`).
  * Simple audit trail logging to a local SQLite database.
  * Counterfactual explanations using `DiCE-ML`.
* **Cut List (Explicitly out of scope for this 9-day sprint):**
  * `ExponentiatedGradient` mitigation (kept only `ThresholdOptimizer` to avoid effort splitting).
  * Formal PDF Annex IV / FRIA documentation (replaced with a concise Regulatory Risk Notes section in the README).
  * Full production web deployment/microservices.

---

## 🏗️ 3. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROJECT ARCHITECTURE & WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────┘
  [Data Ingestion] ➔ [Article 10: Pre-Training Bias Audit] 
                           |
                           ▼
  [Data Pipeline]   ➔ [LightGBM Modeling & Stratified CV] 
                           |
                           ▼
  [Article 13: XAI] ➔ [SHAP (Global Summary & Local Waterfall)] 
                           |
                           ▼
  [Article 9: Risk] ➔ [Fairlearn Mitigation (Threshold Optimization)] 
                           |
                           ▼
  [Article 14]      ➔ [Human Oversight & Uncertainty Escalation Gate] 
                           |
                           ▼
  [Article 12]      ➔ [Audit Trail Logging (JSONL File)] 
                           |
                           ▼
  [Documentation]   ➔ [GitHub README & Trade-off Analysis]
  ```



## 📊 Day 1 Execution & Audit Log (July 25, 2026)

### 1. Environment & Project Initialization
* **IDE & Interpreter:** Configured PyCharm with Python 3.10 virtual environment (`venv`).
* **Version Control:** Initialized Git repository; established folder hierarchy (`data/raw/`, `src/`, `notebooks/`).
* **Dataset:** Downloaded and integrated the **Kaggle Credit Risk Dataset** (32,581 records, 12 features) into `data/raw/credit_risk_dataset.csv`.

### 2. Article 10 Data Governance & Pre-Training Bias Audit
* **Implementation:** Developed and executed `src/audit.py` to evaluate historical demographic bias.
* **Sensitive Attribute:** Age (`person_age < 25` vs. `person_age >= 25`).
* **Audit Results:**
  * Total Dataset Rows: `32,581`
  * Historical Approval Rate (`< 25` years): **76.78%**
  * Historical Approval Rate (`>= 25` years): **79.04%**
  * **Disparate Impact Ratio (DIR): 0.971**
* **Regulatory Verdict:** **PASSED**. The raw historical dataset meets basic Demographic Parity standards ($DIR \ge 0.80$). Pre-training data does not show severe systemic age discrimination.

### 3. Exploratory Data Analysis & Quality Inspection
* **Script:** Executed `src/data_inspection.py` to identify data hygiene issues.
* **Key Findings & Data Cleaning Plan for Day 2:**
  1. **Anomalies / Outliers:** Extreme age value found (`person_age.max() = 144`). Action: Filter out rows where `person_age > 80`.
  2. **Missing Values:** `person_emp_length` (895 missing) and `loan_int_rate` (3,116 missing). Action: Implement median imputation using Scikit-Learn `SimpleImputer`.
  3. **Duplicates:** Identified 165 duplicate entries. Action: Remove duplicates during ingestion.
  4. **Class Imbalance:** Target variable `loan_status` distribution is 78.2% non-default (`0`) vs. 21.8% default (`1`). Action: Utilize stratified splits and `ROC-AUC` / `Gini` metrics rather than accuracy.
