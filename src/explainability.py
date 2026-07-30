import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

from src.modeling import train_final_model

def shap_explainer(model, X_transformed: np.ndarray):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)
    return explainer, shap_values

def plot_global_shap_summary(shap_values, feature_names, save_path = None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names = feature_names, show = False)
    plt.title("Global Feature Importance (SHAP Values)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Global SHAP Plot was saved to {save_path}")
    plt.show()

def log_odds_to_proba(log_odds: float) -> float:
    return 1 / (1 + np.exp(-log_odds))

def generate_adverse_action_notice(shap_values, feature_names, sample_index, save_path = None):
    plt.figure(figsize=(10, 6))
    sample_shap = shap_values[sample_index]
    sample_shap.feature_names = feature_names

    shap.plots.waterfall(sample_shap, show=False)

    plt.title(f"Adverse Action Notice (Local SHAP Explanation for Applicant #{sample_index})", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Local Waterfall plot saved to {save_path}")
    plt.show()


def get_explanation(shap_values, sample_index, feature_names, X_display, reason_map=None, top_n=4):
    sample_shap = shap_values[sample_index]

    base_log_odds = sample_shap.base_values
    final_log_odds = sample_shap.values.sum() + base_log_odds

    base_proba = log_odds_to_proba(base_log_odds)
    final_proba = log_odds_to_proba(final_log_odds)

    # Получаем сырую анкету в виде словаря
    raw_row = X_display.iloc[sample_index].to_dict()

    contributions = []
    for feature, shap_val in zip(feature_names, sample_shap.values):
        real_value = "N/A"

        if feature in raw_row:
            real_value = raw_row[feature]
        else:
            for raw_col in raw_row.keys():
                if feature.startswith(raw_col):
                    real_value = raw_row[raw_col]
                    break

        contributions.append((feature, shap_val, real_value))

    top_features = contributions[:top_n]

    lines = []
    lines.append(
        f"The baseline default risk for an average applicant is {base_proba:.1%}. "
        f"For this specific applicant, the model calculated a default risk of {final_proba:.1%}."
    )
    lines.append("\nThe main factors behind this score were:")

    for feature, shap_val, real_value in top_features:
        direction = "increased" if shap_val > 0 else "decreased"
        strength = "strongly" if abs(shap_val) > 1 else "moderately" if abs(shap_val) > 0.3 else "slightly"
        label = reason_map.get(feature, feature) if reason_map else feature

        lines.append(f"- {label} (value: {real_value}) {strength} {direction} the risk score")

    return "\n".join(lines)

FEATURE_MAPPING = {
    "person_age": "Applicant Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length (years)",
    "loan_amnt": "Requested Loan Amount",
    "loan_percent_income": "Loan-to-Income Ratio",
    "cb_person_cred_hist_length": "Credit History Length",
    "person_home_ownership_RENT": "Home Ownership: Renting",
    "person_home_ownership_OWN": "Home Ownership: Owned",
    "person_home_ownership_MORTGAGE": "Home Ownership: Mortgage",
    "loan_intent_VENTURE": "Loan Purpose: Venture",
    "loan_intent_MEDICAL": "Loan Purpose: Medical",
    "loan_intent_PERSONAL": "Loan Purpose: Personal",
    "loan_intent_EDUCATION": "Loan Purpose: Education",
    "loan_intent_HOMEIMPROVEMENT": "Loan Purpose: Home Improvement",
    "loan_intent_DEBTCONSOLIDATION": "Loan Purpose: Debt Consolidation",
    "cb_person_default_on_file_Y": "Historical Default: Yes",
    "cb_person_default_on_file_N": "Historical Default: No"
}


def get_waterfall_figure(shap_values, feature_names, sample_index=0):
    sample_shap = shap_values[sample_index]
    sample_shap.feature_names = feature_names

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sample_shap, show=False)
    plt.title(f"Adverse Action Notice (Applicant #{sample_index})",
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()

    return plt.gcf()