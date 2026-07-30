import pandas as pd
from src.modeling import load_final_model
from src.fairness import load_thresholds
from src.oversight import process_new_applicant
from src.oversight import get_risk_for_new_applicant

final_model, final_preprocessor = load_final_model()
group_thresholds = load_thresholds()

# Applicant 1 - low risk
applicant_low_risk = pd.DataFrame([{
    "person_age": 35,
    "person_income": 95000,
    "person_emp_length": 8.0,
    "loan_amnt": 5000,
    "loan_percent_income": 0.05,
    "cb_person_cred_hist_length": 10,
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": "N"
}])

# Applicant 2 - high risk
applicant_high_risk = pd.DataFrame([{
    "person_age": 22,
    "person_income": 20000,
    "person_emp_length": 0.5,
    "loan_amnt": 15000,
    "loan_percent_income": 0.75,
    "cb_person_cred_hist_length": 2,
    "person_home_ownership": "RENT",
    "loan_intent": "MEDICAL",
    "cb_person_default_on_file": "Y"
}])

# Applicant 3 - borderline(not very high not very low risk)
applicant_borderline = pd.DataFrame([{
    "person_age": 28,
    "person_income": 45000,
    "person_emp_length": 3.0,
    "loan_amnt": 10000,
    "loan_percent_income": 0.22,
    "cb_person_cred_hist_length": 5,
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "DEBTCONSOLIDATION",
    "cb_person_default_on_file": "N"
}])

applicant_very_safe = pd.DataFrame([{
    "person_age": 45,
    "person_income": 150000,
    "person_emp_length": 20.0,
    "loan_amnt": 2000,
    "loan_percent_income": 0.01,
    "cb_person_cred_hist_length": 20,
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": "N"
}])

test_applicants = [
    ("Low Risk Applicant", applicant_low_risk),
    ("High Risk Applicant", applicant_high_risk),
    ("Borderline Applicant", applicant_borderline),
    ("Very Safe Applicant", applicant_very_safe)
]

for name, applicant in test_applicants:
    home_group = applicant["person_home_ownership"].iloc[0]

    risk_proba, decision = process_new_applicant(
        applicant, final_model, final_preprocessor,
        group_thresholds, home_group
    )

    print(f"{name}: risk={risk_proba:.4f} | group={home_group} | decision={decision}")