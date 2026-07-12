"""
hyperparameter_tuning.py
------------------------
Handles GridSearchCV and RandomizedSearchCV for all estimators.
"""

import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from configs.config import TUNING_GRIDS, RANDOM_STATE

logger = logging.getLogger(__name__)

def get_grid_for_model(model_name: str) -> Dict[str, Any]:
    """Retrieves the hyperparameter tuning grid for a specific model."""
    name = model_name.lower().replace("_", "").replace("-", "")
    if name in ("randomforest", "rf"):
        return TUNING_GRIDS["random_forest"]
    elif name in ("xgboost", "xgb"):
        return TUNING_GRIDS["xgboost"]
    elif name in ("lightgbm", "lgb"):
        return TUNING_GRIDS["lightgbm"]
    elif name in ("catboost", "cb"):
        return TUNING_GRIDS["catboost"]
    elif name in ("logisticregression", "lr"):
        return TUNING_GRIDS["logistic_regression"]
    elif name in ("decisiontree", "dt"):
        return TUNING_GRIDS["decision_tree"]
    else:
        raise ValueError(f"No search grid configured for model name: {model_name}")

def tune_model(
    model_name: str,
    base_estimator: Any,
    X: Any,
    y: Any,
    search_type: str = "grid",
    cv: int = 5,
    scoring: str = "accuracy",
    n_iter: int = 10,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """Runs hyperparameter search on the base estimator.
    
    Returns:
        best_estimator: The fitted estimator with the best found parameters.
        best_params: A dictionary of the best parameters.
    """
    param_grid = get_grid_for_model(model_name)
    logger.info(f"Tuning {model_name} using {search_type} search. Grid size: {param_grid}")

    if search_type.lower() == "grid":
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            **kwargs
        )
    elif search_type.lower() == "random":
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")

    search.fit(X, y)
    logger.info(f"Best parameters found for {model_name}: {search.best_params_} (Best CV Score: {search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_
