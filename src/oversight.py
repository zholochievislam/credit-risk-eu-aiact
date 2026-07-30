import os
from src.modeling import load_final_model
import json
from datetime import datetime

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_CURRENT_DIR, "..", "logs")

def route_decision(risk_proba, group_threshold, uncertainty_margin = 0.10):
    lower_bound = group_threshold - uncertainty_margin
    upper_bound = group_threshold + uncertainty_margin

    if risk_proba < lower_bound:
        return "AUTO_APPROVE"
    elif risk_proba > upper_bound:
        return "AUTO_REJECT"
    else:
        return ("MANUAL_REVIEW")

def get_risk_for_new_applicant(applicant_data, final_model, final_preprocessor):
    applicant_transformed = final_preprocessor.transform(applicant_data)
    risk_proba = final_model.predict_proba(applicant_transformed)[:, 1][0]

    return risk_proba

def process_new_applicant(applicant_data, final_model, final_preprocessor, group_thresholds, home_ownership_group):
    risk_proba = get_risk_for_new_applicant(applicant_data, final_model, final_preprocessor)
    group_threshold = group_thresholds[home_ownership_group]
    decision = route_decision(risk_proba, group_threshold)

    log_decision(applicant_data, risk_proba, decision, group_threshold, home_ownership_group)

    return risk_proba, decision

def log_decision(applicant_data, risk_proba, decision, group_threshold, home_ownership_group, log_path="../logs/decisions_log.jsonl"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "applicant_data": applicant_data.to_dict(orient="records")[0],
        "home_ownership_group": home_ownership_group,
        "risk_proba": float(risk_proba),
        "group_threshold": float(group_threshold),
        "decision": decision
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"Decision logged to {log_path}")
