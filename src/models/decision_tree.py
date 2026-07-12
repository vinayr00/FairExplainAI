"""Decision Tree Classifier model definition."""
from sklearn.tree import DecisionTreeClassifier

def get_decision_tree_model(random_state=42, **kwargs):
    """Returns a configured DecisionTreeClassifier instance."""
    return DecisionTreeClassifier(random_state=random_state, **kwargs)
