"""Random Forest Classifier model definition."""
from sklearn.ensemble import RandomForestClassifier

def get_random_forest_model(random_state=42, **kwargs):
    """Returns a configured RandomForestClassifier instance."""
    return RandomForestClassifier(random_state=random_state, **kwargs)
