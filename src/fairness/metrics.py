"""Fairness metrics computation."""
import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    """Computes basic performance and fairness metrics across sensitive groups."""
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    return metric_frame

def get_fairness_summary(y_true, y_pred, sensitive_features) -> dict:
    """Computes a summary dictionary of standard fairness difference and ratio metrics.
    
    Returns:
        A dictionary containing:
            - demographic_parity_difference
            - equalized_odds_difference
            - equal_opportunity_difference
            - disparate_impact_ratio
    """
    mf = compute_fairness_metrics(y_true, y_pred, sensitive_features)
    
    # 1. Demographic Parity Difference: max difference in selection rate
    dp_diff = mf.difference()["selection_rate"]
    
    # 2. Equalized Odds Difference: max of TPR and FPR differences
    tpr_diff = mf.difference()["tpr"]
    fpr_diff = mf.difference()["fpr"]
    eo_diff = max(tpr_diff, fpr_diff)
    
    # 3. Equal Opportunity Difference: TPR difference
    eop_diff = tpr_diff
    
    # 4. Disparate Impact Ratio: selection rate ratio (min / max)
    # Avoid division by zero
    sel_rates = mf.by_group["selection_rate"]
    if len(sel_rates) > 1:
        min_rate = float(sel_rates.min())
        max_rate = float(sel_rates.max())
        di_ratio = min_rate / max_rate if max_rate > 0 else 1.0
    else:
        di_ratio = 1.0
        
    return {
        "demographic_parity_difference": float(dp_diff),
        "equalized_odds_difference": float(eo_diff),
        "equal_opportunity_difference": float(eop_diff),
        "disparate_impact_ratio": float(di_ratio)
    }

