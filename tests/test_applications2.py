import pandas as pd
from src.modeling import load_final_model
from src.fairness import load_thresholds
from src.oversight import process_new_applicant


def test_three_risk_profiles():
    final_model, final_preprocessor = load_final_model()
    group_thresholds = load_thresholds()

    applicant_moderate = pd.DataFrame([{
        "person_age": 35, "person_income": 95000, "person_emp_length": 8.0,
        "loan_amnt": 5000, "loan_percent_income": 0.05,
        "cb_person_cred_hist_length": 10, "person_home_ownership": "MORTGAGE",
        "loan_intent": "PERSONAL", "cb_person_default_on_file": "N"
    }])

    applicant_high_risk = pd.DataFrame([{
        "person_age": 22, "person_income": 20000, "person_emp_length": 0.5,
        "loan_amnt": 15000, "loan_percent_income": 0.75,
        "cb_person_cred_hist_length": 2, "person_home_ownership": "RENT",
        "loan_intent": "MEDICAL", "cb_person_default_on_file": "Y"
    }])

    applicant_very_safe = pd.DataFrame([{
        "person_age": 45, "person_income": 150000, "person_emp_length": 20.0,
        "loan_amnt": 2000, "loan_percent_income": 0.01,
        "cb_person_cred_hist_length": 20, "person_home_ownership": "MORTGAGE",
        "loan_intent": "PERSONAL", "cb_person_default_on_file": "N"
    }])

    risk_moderate, decision_moderate = process_new_applicant(
        applicant_moderate, final_model, final_preprocessor,
        group_thresholds, "MORTGAGE"
    )
    risk_high, decision_high = process_new_applicant(
        applicant_high_risk, final_model, final_preprocessor,
        group_thresholds, "RENT"
    )
    risk_safe, decision_safe = process_new_applicant(
        applicant_very_safe, final_model, final_preprocessor,
        group_thresholds, "MORTGAGE"
    )

    print(f"Moderate applicant: risk={risk_moderate:.3f}, decision={decision_moderate}")
    print(f"High risk applicant: risk={risk_high:.3f}, decision={decision_high}")
    print(f"Very safe applicant: risk={risk_safe:.3f}, decision={decision_safe}")

    assert decision_moderate == "MANUAL_REVIEW"
    assert decision_high == "AUTO_REJECT"
    assert decision_safe == "AUTO_APPROVE"