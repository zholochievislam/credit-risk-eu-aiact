import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import cross_val_predict


from src.data_processing import clean_raw_data,get_preprocessor

def stratified_cv(df: pd.DataFrame, num_cols: list, cat_cols: list,n_splits=5, return_oof: bool = False, verbose: bool=False):
    X = df[num_cols + cat_cols]
    y = df["loan_status"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    roc_aucs, ginis, pr_aucs = [], [], []

    oof_proba = np.zeros(len(df)) if return_oof else None

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        preprocessor = get_preprocessor(num_cols, cat_cols)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        model = LGBMClassifier(n_estimators=100, learning_rate=0.01,verbose= -1, random_state=42)
        model.fit(X_train_trans, y_train)

        y_pred_prob = model.predict_proba(X_test_trans)[:,1]

        if return_oof:
            oof_proba[test_index] = y_pred_prob

        roc_auc = roc_auc_score(y_test, y_pred_prob)
        pr_auc = average_precision_score(y_test, y_pred_prob)
        gini = 2*roc_auc -1

        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        ginis.append(gini)

        if verbose:
            print(f"Fold {fold}: ROC-AUC = {roc_auc:.4f} | Gini = {gini:.4f} | PR-AUC = {pr_auc:.4f}")

    metrics = {
        "mean_roc_auc": np.mean(roc_aucs),
        "mean_gini": np.mean(ginis),
        "mean_pr_auc": np.mean(pr_aucs)
    }

    if return_oof:
        oof_series = pd.Series(oof_proba, index=df.index, name="oof_proba")
        return metrics, oof_series

    return metrics

def train_final_model(df: pd.DataFrame, num_cols: list, cat_cols: list):
    X = df[num_cols + cat_cols]
    y = df["loan_status"]

    preprocessor = get_preprocessor(num_cols, cat_cols)
    X_trans = preprocessor.fit_transform(X)

    model = LGBMClassifier(n_estimators=100, learning_rate=0.01,verbose= -1, random_state=42)
    model.fit(X_trans, y)

    return model, preprocessor


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_CURRENT_DIR, "..", "models")


def save_model(model, preprocessor, model_path=None, preprocessor_path=None):
    if model_path is None:
        model_path = os.path.join(_MODELS_DIR, "final_model.joblib")
    if preprocessor_path is None:
        preprocessor_path = os.path.join(_MODELS_DIR, "final_preprocessor.joblib")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")


def load_final_model(model_path=None, preprocessor_path=None):
    if model_path is None:
        model_path = os.path.join(_MODELS_DIR, "final_model.joblib")
    if preprocessor_path is None:
        preprocessor_path = os.path.join(_MODELS_DIR, "final_preprocessor.joblib")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

if __name__ == "__main__":
    filepath = "../data/raw/credit_risk_dataset.csv"
    try:
        raw_data = pd.read_csv(filepath)
        cleaned_data = clean_raw_data(raw_data)
    except FileNotFoundError:
        print("File not found")

    # Model A: baseline model

    num_cols_a = ["person_age", "person_income", "person_emp_length", "loan_amnt","loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    cat_cols_a = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

    print("__"*20)
    print("Model A: Baseline(with leakage features: loan_grade + loan_int_rate)")
    print("__"*20)
    metrics_a = stratified_cv(cleaned_data, num_cols_a, cat_cols_a)
    print(f"Model A CV Average -> ROC-AUC: {metrics_a['mean_roc_auc']:.4f}, Gini: {metrics_a['mean_gini']:.4f}, PR-AUC: {metrics_a['mean_pr_auc']:.4f}")

    # Model B: production candidate, without leakage(loan_int_rate and loan_grade are dropped)
    num_cols_b = ["person_age", "person_income", "person_emp_length", "loan_amnt","loan_percent_income", "cb_person_cred_hist_length"]
    cat_cols_b = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]

    print("\n")
    print("__"*20)
    print("MODEL B: Production Candidate (No leakage)")
    print("__" * 20)
    metrics_b = stratified_cv(cleaned_data, num_cols_b, cat_cols_b)
    print(f"Model B CV Average -> ROC-AUC: {metrics_b['mean_roc_auc']:.4f}, Gini: {metrics_b['mean_gini']:.4f}, PR-AUC: {metrics_b['mean_pr_auc']:.4f}")

    print("\nTraining final production model (Model B) on 100% of dataset")
    final_model, final_preprocessor = train_final_model(cleaned_data, num_cols_b, cat_cols_b)
    print("Final model training finished.")

    save_model(final_model, final_preprocessor)


