"""Pipeline class to combine model training with fairness mitigations."""
import logging
from src.fairness.mitigation import mitigate_bias_reductions
from src.fairness.threshold_optimizer import optimize_threshold

logger = logging.getLogger(__name__)

class FairLearnPipeline:
    """Manages the lifecycle of standard estimators with fairness mitigation overrides."""

    def __init__(self, base_estimator, mitigation_method=None, constraint="demographic_parity"):
        self.base_estimator = base_estimator
        self.mitigation_method = mitigation_method
        self.constraint = constraint
        self.model = None

    def fit(self, X, y, sensitive_features):
        """Fits the pipeline with or without bias mitigation."""
        if self.mitigation_method is None:
            logger.info("Fitting base estimator without fairness mitigation.")
            self.base_estimator.fit(X, y)
            self.model = self.base_estimator
        elif self.mitigation_method == "reductions":
            logger.info(f"Fitting reduction model with constraint: {self.constraint}")
            self.model = mitigate_bias_reductions(
                self.base_estimator, X, y, sensitive_features, constraint_type=self.constraint
            )
        elif self.mitigation_method == "postprocessing":
            logger.info(f"Fitting postprocessing model with constraint: {self.constraint}")
            self.model = optimize_threshold(
                self.base_estimator, X, y, sensitive_features, constraint=self.constraint
            )
        else:
            raise ValueError(f"Unknown mitigation method: {self.mitigation_method}")
        return self

    def predict(self, X, sensitive_features=None):
        """Generates predictions from the fitted model."""
        if self.model is None:
            raise ValueError("Pipeline is not fitted yet.")

        if self.mitigation_method == "postprocessing":
            return self.model.predict(X, sensitive_features=sensitive_features)
        return self.model.predict(X)
