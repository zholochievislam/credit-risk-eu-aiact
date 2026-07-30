import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from sklearn.model_selection import train_test_split
from fairlearn.postprocessing import ThresholdOptimizer


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

        rejection_rate = mf.by_group['selection_rate']
        approval_rate = 1 - rejection_rate

        tpr_ = mf.by_group['true_positive_rate']

        max_appr = approval_rate.max()
        dir_metric = approval_rate.min() / max_appr if max_appr > 0 else 0.0

        eod_metric = tpr_.max() - tpr_.min()

        group_sizes = sensitive_features[group_col].value_counts()

        audit_results[group_col] = {
            'metric_frame': mf,
            'approval_rate': approval_rate,
            'DIR': dir_metric,
            'EOD': eod_metric,
            'group_sizes': group_sizes
        }
    return audit_results

def print_fairness_audit(audit_results):
    print("__" * 20)
    print("AI FAIRNESS AUDIT REPORT")
    print("__" * 20)

    for group_col, data in audit_results.items():
        print(f"\n--- Attribute: {group_col.upper()} ---")

        print("\nDetailed Metrics by Group:")
        print(data['metric_frame'].by_group.to_string())

        summary_table = pd.DataFrame({
            'approval_rate': data['approval_rate'],
            'group_size': data['group_sizes']
        })
        print("\nApproval Rate & Group Size:")
        print(summary_table.to_string())

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


def get_thresholds_per_group(y_true, y_pred_proba, sensitive_series):
    thresholds = {}

    for group_name in sensitive_series.unique():
        mask = (sensitive_series == group_name)

        group_y_true = y_true[mask]
        group_y_proba = y_pred_proba[mask]

        group_threshold = get_threshold(group_y_true, group_y_proba)
        thresholds[group_name] = group_threshold

    return thresholds

def apply_group_thresholds(y_pred_proba, sensitive_series, thresholds):
    y_pred = pd.Series(index=sensitive_series.index, dtype=int)

    for group_name, group_threshold in thresholds.items():
        mask = (sensitive_series == group_name)
        y_pred[mask] = (y_pred_proba[mask] >= group_threshold).astype(int)

    return y_pred

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_CURRENT_DIR, "..", "models")


def save_thresholds(thresholds: dict, filepath: str = None):
    if filepath is None:
        filepath = os.path.join(_MODELS_DIR, "group_thresholds.json")

    clean_thresholds = {k: float(v) for k, v in thresholds.items()}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(clean_thresholds, f, indent=2)

    print(f"Thresholds saved to {filepath}")


def load_thresholds(filepath: str = None) -> dict:
    if filepath is None:
        filepath = os.path.join(_MODELS_DIR, "group_thresholds.json")

    with open(filepath, "r") as f:
        return json.load(f)

def plot_fairness_tradeoff(acc_before, acc_after, eod_before, eod_after, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left graph: Accuracy
    axes[0].bar(['Before', 'After'], [acc_before, acc_after], color=['#d62728', '#2ca02c'])
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([acc_before, acc_after]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    # Right graph: EOD (more less - more better)
    axes[1].bar(['Before', 'After'], [eod_before, eod_after], color=['#d62728', '#2ca02c'])
    axes[1].set_title('Equal Opportunity Difference (lower = fairer)')
    axes[1].set_ylim(0, max(eod_before, eod_after) + 0.1)
    for i, v in enumerate([eod_before, eod_after]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

    plt.suptitle('Fairness Mitigation Trade-off: person_home_ownership', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trade-off plot saved to {save_path}")
    plt.show()

def save_metrics(metrics: dict, filepath: str = None):
    if filepath is None:
        filepath = os.path.join(_MODELS_DIR, "model_metrics.json")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str = None) -> dict:
    if filepath is None:
        filepath = os.path.join(_MODELS_DIR, "model_metrics.json")

    with open(filepath, "r") as f:
        return json.load(f)