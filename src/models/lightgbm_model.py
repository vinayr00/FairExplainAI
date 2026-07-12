"""LightGBM Classifier model definition."""

def get_lightgbm_model(random_state=42, **kwargs):
    """Returns a configured LGBMClassifier instance."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=random_state, **kwargs)
    except ImportError:
        raise ImportError("lightgbm is not installed. Please install it using `pip install lightgbm`.")
