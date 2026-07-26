import pandas as pd
import numpy as np

def run_pre_training_bias_audit(df: pd.DataFrame, sensitive_col: str = 'person_age', target_col: str = 'loan_status'):
    df['is_young'] = (df[sensitive_col] < 25).astype(int) # if age<25 then 1, otherwise 0
    df['approved'] = (df[target_col] == 0).astype(int)  # where 1 - success/approval, and 0 - reject, default

    # Average percentage of approval:
    approval_rates = df.groupby('is_young')['approved'].mean()

    young_approval = approval_rates[1]  # approval rate for age < 25y
    older_approval = approval_rates[0]  # approval rate for age >= 25y

    # Disparate Impact Ratio(DIR):
    disparate_impact = young_approval / older_approval

    print("-" * 15)
    print("EU AI ACT: ARTICLE 10 PRE-TRAINING DATA BIAS AUDIT")
    print("-" * 15)

    print(f"Historical Approval Rate (<25):  {young_approval:.2%}")
    print(f"Historical Approval Rate (>=25): {older_approval:.2%}")
    print(f"Disparate Impact Ratio (DIR):    {disparate_impact:.3f}")
    print("-" * 15)

    if disparate_impact < 0.80:
        print("Disparate impact detected against younger borrowers")
        print("ACTION REQUIRED: Article 10 compliance requires mitigation during modeling (Article 9).")
    else:
        print("RESULT:Historical data meets baseline Demographic Parity standards.")

if __name__ == '__main__':
    filepath = "/Users/islam/Desktop/Budapest_Bootcamp/data/raw/credit_risk_dataset.csv"
    try:
        data = pd.read_csv(filepath)
        run_pre_training_bias_audit(data)
    except FileNotFoundError:
        print("File not found")
