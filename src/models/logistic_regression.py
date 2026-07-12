"""Logistic Regression model definition."""
from sklearn.linear_model import LogisticRegression

def get_logistic_regression_model(random_state=42, max_iter=1000, **kwargs):
    """Returns a configured LogisticRegression instance."""
    return LogisticRegression(random_state=random_state, max_iter=max_iter, **kwargs)
