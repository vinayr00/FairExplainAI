"""Bias mitigation algorithms wrappers."""
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

def mitigate_bias_reductions(estimator, X, y, sensitive_features, constraint_type="demographic_parity"):
    """Mitigates bias using in-processing reduction algorithms (Exponentiated Gradient)."""
    if constraint_type == "demographic_parity":
        constraint = DemographicParity()
    elif constraint_type == "equalized_odds":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")
        
    mitigator = ExponentiatedGradient(estimator, constraint)
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    return mitigator
