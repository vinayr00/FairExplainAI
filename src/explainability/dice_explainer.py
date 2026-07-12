"""DiCE Counterfactual Explainer helper functions."""

def generate_counterfactuals(
    model,
    query_instances,
    data_interface,
    outcome_name,
    continuous_features=None,
    total_CFs=4,
):
    """Generates counterfactual explanations using DiCE library."""
    try:
        import dice_ml
        
        if continuous_features is None:
            # Auto-detect numeric columns as continuous features (excluding the target outcome)
            continuous_features = [
                col for col in data_interface.select_dtypes(include=["number"]).columns
                if col != outcome_name
            ]
        
        import pandas as pd

        # Identify numeric categorical columns that need casting to prevent LossySetitemError in Pandas 2.x
        cols_to_cast = {}
        for col in data_interface.columns:
            if col != outcome_name and col not in continuous_features:
                if pd.api.types.is_numeric_dtype(data_interface[col]):
                    cols_to_cast[col] = data_interface[col].dtype

        if cols_to_cast:
            data_interface = data_interface.copy()
            query_instances = query_instances.copy()
            for col, dtype in cols_to_cast.items():
                data_interface[col] = data_interface[col].astype(str)
                if col in query_instances.columns:
                    query_instances[col] = query_instances[col].astype(str)

            class ModelWrapper:
                def __init__(self, inner_model, cols_to_cast):
                    self.inner_model = inner_model
                    self.cols_to_cast = cols_to_cast

                def _prepare_input(self, X):
                    X_copy = X.copy()
                    for col, dtype in self.cols_to_cast.items():
                        if col in X_copy.columns:
                            X_copy[col] = X_copy[col].astype(dtype)
                    return X_copy

                def predict(self, X, **kwargs):
                    return self.inner_model.predict(self._prepare_input(X), **kwargs)

                def predict_proba(self, X, **kwargs):
                    return self.inner_model.predict_proba(self._prepare_input(X), **kwargs)

                def __getattr__(self, name):
                    return getattr(self.inner_model, name)

            model = ModelWrapper(model, cols_to_cast)

        # Create a DiCE model wrapper
        m = dice_ml.Model(model=model, backend="sklearn")
        # Create a DiCE data interface
        d = dice_ml.Data(
            dataframe=data_interface,
            continuous_features=continuous_features,
            outcome_name=outcome_name,
        )
        
        exp = dice_ml.Dice(d, m, method="random")
        dice_exp = exp.generate_counterfactuals(
            query_instances, total_CFs=total_CFs, desired_class="opposite"
        )
        return dice_exp
    except ImportError:
        raise ImportError("dice_ml is not installed. Please install it using `pip install dice-ml`.")

