"""Unit tests for fairness modules."""
import numpy as np
import pytest
from src.fairness.metrics import compute_fairness_metrics, get_fairness_summary

def test_compute_fairness_metrics():
    """Verifies demographic and performance metrics generation."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    
    mf = compute_fairness_metrics(y_true, y_pred, sensitive)
    assert mf is not None
    # Check that accuracy is calculated in results
    assert "accuracy" in mf.overall.index

def test_get_fairness_summary():
    """Verifies that the fairness metrics summary dictionary is generated correctly."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    
    summary = get_fairness_summary(y_true, y_pred, sensitive)
    assert "demographic_parity_difference" in summary
    assert "equalized_odds_difference" in summary
    assert "equal_opportunity_difference" in summary
    assert "disparate_impact_ratio" in summary
    assert isinstance(summary["demographic_parity_difference"], float)

