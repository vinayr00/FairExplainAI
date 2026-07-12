"""Post-processing threshold optimizer."""
from fairlearn.postprocessing import ThresholdOptimizer

class Float64PredictProbaWrapper:
    """Wrapper to force predict_proba outputs to float64, preventing pandas 2.x strict-casting errors."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return self.estimator.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        res = self.estimator.predict_proba(X, **kwargs)
        return res.astype(float)

    def get_params(self, deep=True):
        return {"estimator": self.estimator}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_clone__(self):
        from sklearn.base import clone
        return Float64PredictProbaWrapper(clone(self.estimator))

    def __getattr__(self, name):
        if name == "estimator":
            raise AttributeError("estimator attribute not initialized yet")
        if "estimator" not in self.__dict__:
            raise AttributeError("estimator attribute not initialized yet")
        return getattr(self.estimator, name)

def optimize_threshold(estimator, X, y, sensitive_features, constraint="demographic_parity"):
    """Applies post-processing ThresholdOptimizer to enforce fairness constraints."""
    wrapped_estimator = Float64PredictProbaWrapper(estimator)
    optimizer = ThresholdOptimizer(
        estimator=wrapped_estimator,
        constraints=constraint,
        predict_method="predict_proba"
    )
    optimizer.fit(X, y, sensitive_features=sensitive_features)
    return optimizer
