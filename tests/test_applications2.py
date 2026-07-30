import pandas as pd
from src.modeling import load_final_model
from src.fairness import load_thresholds
from src.oversight import process_new_applicant

def test_three_risk_profiles():
    final_model, final_preprocessor = load_final_model()
    group_thresholds = load_thresholds()

    applicant_low_risk = pd.DataFrame([{
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

    risk_low, decision_low = process_new_applicant(
        applicant_low_risk, final_model, final_preprocessor,
        group_thresholds, "MORTGAGE"
    )
    risk_high, decision_high = process_new_applicant(
        applicant_high_risk, final_model, final_preprocessor,
        group_thresholds, "RENT"
    )

    print(f"Low risk applicant: risk={risk_low:.3f}, decision={decision_low}")
    print(f"High risk applicant: risk={risk_high:.3f}, decision={decision_high}")

    # Проверяем, что логика сработала правильно, а не просто "не упала"
    assert decision_low == "AUTO_APPROVE"
    assert decision_high == "AUTO_REJECT"