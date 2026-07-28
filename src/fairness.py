import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate


def get_threshold(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = (tpr - fpr).argmax()
    best_threshold = thresholds[optimal_idx]

    return best_threshold

def fairness_data(X_raw):
    df_fair = X_raw.copy()
    df_fair['age_group'] = pd.cut(
        df_fair['person_age'],
        bins=[17, 25, 40, 100],
        labels=['young (18-25)', 'middle (26-40)', 'older (40+)']
    )

    return df_fair

def run_fairness_audit(y_true, y_pred, sensitive_features, group_cols):
    metrics = {'selection_rate': selection_rate,'true_positive_rate': true_positive_rate}

    audit_results = {}

    for group_col in group_cols:
        mf = MetricFrame(metrics=metrics,y_true=y_true,y_pred=y_pred,sensitive_features=sensitive_features[group_col])


        sel = mf.by_group['selection_rate']
        tpr_ = mf.by_group['true_positive_rate']

        max_sel = sel.max()
        dir_metric = sel.min() / max_sel if max_sel > 0 else 0.0

        eod_metric = tpr_.max() - tpr_.min()

        audit_results[group_col] = {
            'metric_frame': mf,
            'DIR': dir_metric,
            'EOD': eod_metric
        }

    return audit_results

def print_fairness_audit(audit_results):
    print("__" * 20)
    print("⚖️ AI FAIRNESS AUDIT REPORT")
    print("__" * 20)

    for group_col, data in audit_results.items():
        print(f"\n--- Attribute: {group_col.upper()} ---")

        print("\nDetailed Metrics by Group:")
        print(data['metric_frame'].by_group.to_string())

        dir_metric = data['DIR']
        eod_metric = data['EOD']

        print("\nHeadline Fairness Indicators:")
        print(f" Disparate Impact Ratio (DIR): {dir_metric:.2f}")
        print(f" Equal Opportunity Diff (EOD): {eod_metric:.2f}")

        # Проверяем "Правило 80%"
        if dir_metric < 0.80:
            print(" FLAG: DIR is below 0.80 (Potential Disparate Impact)")
        else:
            print(" PASS: DIR is acceptable (>= 0.80)")

        if eod_metric > 0.10:
            print(" FLAG: EOD is above 0.10 (Unequal prediction errors)")
        else:
            print(" PASS: EOD is acceptable (<= 0.10)")
    print("\n")
    print("__" * 25)
