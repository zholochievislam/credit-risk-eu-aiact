import pandas as pd
from src.data_processing import clean_raw_data
from src.modeling import stratified_cv

BASE_NUM_COLS = ["person_age", "person_income", "person_emp_length", "loan_amnt","loan_percent_income", "cb_person_cred_hist_length"]
BASE_CAT_COLS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]

# (label, extra numeric cols to add back, extra categorical cols to add back)
EXPERIMENTS = [
    ("All features (with loan_grade + loan_int_rate)", ["loan_int_rate"], ["loan_grade"]),
    ("Without loan_int_rate only", [], ["loan_grade"]),
    ("Without loan_grade only", ["loan_int_rate"], []),
    ("Without both (production candidate)", [], []),
]


def run_leakage_comparison(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    results = []
    for label, extra_num, extra_cat in EXPERIMENTS:
        num_cols = BASE_NUM_COLS + extra_num
        cat_cols = BASE_CAT_COLS + extra_cat

        print(f"Running: {label}")
        metrics = stratified_cv(df, num_cols, cat_cols, n_splits=n_splits, verbose=False)
        results.append({
            "model": label,
            "mean_roc_auc": round(metrics["mean_roc_auc"], 4),
            "mean_gini": round(metrics["mean_gini"], 4),
            "mean_pr_auc": round(metrics["mean_pr_auc"], 4),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    filepath = "../data/raw/credit_risk_dataset.csv"
    try:
        raw_data = pd.read_csv(filepath)
        cleaned_data = clean_raw_data(raw_data)
    except FileNotFoundError:
        print("File not found")
        raise SystemExit

    comparison_table = run_leakage_comparison(cleaned_data)

    print("\n")
    print("__"*20)
    print("LEAKAGE COMPARISON — CV AVERAGE METRICS")
    print("__"*20)
    print(comparison_table.to_string(index=False))