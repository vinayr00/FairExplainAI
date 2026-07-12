"""Factory module for generating models."""
from src.models.random_forest import get_random_forest_model
from src.models.xgboost_model import get_xgboost_model
from src.models.lightgbm_model import get_lightgbm_model
from src.models.catboost_model import get_catboost_model
from src.models.logistic_regression import get_logistic_regression_model
from src.models.decision_tree import get_decision_tree_model

class ModelFactory:
    """Model factory to construct machine learning classifiers."""

    @staticmethod
    def get_model(model_name: str, random_state: int = 42, **kwargs):
        """Builds and returns model instances based on model_name string."""
        name = model_name.lower().replace("_", "").replace("-", "")
        if name in ("randomforest", "rf"):
            return get_random_forest_model(random_state=random_state, **kwargs)
        elif name in ("xgboost", "xgb"):
            return get_xgboost_model(random_state=random_state, **kwargs)
        elif name in ("lightgbm", "lgb"):
            return get_lightgbm_model(random_state=random_state, **kwargs)
        elif name in ("catboost", "cb"):
            return get_catboost_model(random_state=random_state, **kwargs)
        elif name in ("logisticregression", "lr"):
            return get_logistic_regression_model(random_state=random_state, **kwargs)
        elif name in ("decisiontree", "dt"):
            return get_decision_tree_model(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

