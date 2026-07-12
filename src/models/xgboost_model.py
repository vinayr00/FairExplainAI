"""XGBoost Classifier model definition."""

def get_xgboost_model(random_state=42, **kwargs):
    """Returns a configured XGBClassifier instance."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state, **kwargs)
    except ImportError:
        raise ImportError("xgboost is not installed. Please install it using `pip install xgboost`.")
