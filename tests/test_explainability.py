"""Unit tests for explainability modules."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.explainability.dice_explainer import generate_counterfactuals

def test_dice_explainer():
    """Verifies that dice_explainer generates counterfactual examples correctly."""
    # Create simple dummy training dataset
    data = pd.DataFrame({
        "age": [22, 25, 47, 52, 36, 29, 31, 44],
        "priors": [0, 1, 4, 3, 2, 0, 1, 5],
        "outcome": [0, 0, 1, 1, 0, 0, 0, 1]
    })
    
    X = data[["age", "priors"]]
    y = data["outcome"]
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Instance to query
    query = pd.DataFrame({"age": [22], "priors": [0]})
    
    dice_exp = generate_counterfactuals(
        model=model,
        query_instances=query,
        data_interface=data,
        outcome_name="outcome",
        continuous_features=["age", "priors"],
        total_CFs=2
    )
    
    assert dice_exp is not None
    assert dice_exp.cf_examples_list is not None
    assert len(dice_exp.cf_examples_list) > 0
