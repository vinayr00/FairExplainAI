"""CatBoost Classifier model definition."""

def get_catboost_model(random_state=42, **kwargs):
    """Returns a configured CatBoostClassifier instance."""
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_state=random_state, verbose=0, **kwargs)
    except ImportError:
        raise ImportError("catboost is not installed. Please install it using `pip install catboost`.")
