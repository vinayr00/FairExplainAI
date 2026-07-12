"""Unit tests for models factory and hyperparameter tuning."""
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.models.model_factory import ModelFactory
from src.models.hyperparameter_tuning import tune_model

def test_model_factory_retrieval():
    """Verifies that ModelFactory correctly instantiates estimators."""
    model = ModelFactory.get_model("random_forest")
    assert isinstance(model, RandomForestClassifier)
    
    model_dt = ModelFactory.get_model("decision_tree")
    assert isinstance(model_dt, DecisionTreeClassifier)
    
    model_cat = ModelFactory.get_model("catboost")
    from catboost import CatBoostClassifier
    assert isinstance(model_cat, CatBoostClassifier)
    
    with pytest.raises(ValueError):
        ModelFactory.get_model("invalid_model_name")

def test_hyperparameter_tuning():
    """Verifies that hyperparameter tuning runs correctly on dummy data."""
    X = pd.DataFrame({"feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "feat2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    base_model = ModelFactory.get_model("decision_tree")
    best_estimator, best_params = tune_model(
        model_name="decision_tree",
        base_estimator=base_model,
        X=X,
        y=y,
        search_type="grid",
        cv=2
    )
    
    assert best_estimator is not None
    assert "max_depth" in best_params
    assert isinstance(best_estimator, DecisionTreeClassifier)


