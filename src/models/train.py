"""Training pipeline for models."""
import logging

logger = logging.getLogger(__name__)

def train_model(model, X_train, y_train, **kwargs):
    """Fits the model on training data."""
    logger.info(f"Training model: {type(model).__name__}")
    model.fit(X_train, y_train, **kwargs)
    return model
