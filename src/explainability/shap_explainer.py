"""SHAP Explainer helper functions."""
import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_values(model, X):
    """Computes SHAP values for a given model and input features.
    
    Tries TreeExplainer first (for trees), then falls back to LinearExplainer
    or KernelExplainer/Explainer.
    """
    try:
        # Check if it has tree structure
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
    except Exception:
        try:
            # Fallback to KernelExplainer wrapping predict method
            predict_fn = getattr(model, "predict_proba", model.predict)
            explainer = shap.KernelExplainer(predict_fn, shap.kmeans(X, 10))
            shap_values = explainer.shap_values(X)
        except Exception:
            # Absolute fallback
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
    return shap_values

def plot_shap_summary(shap_values, X, save_path=None):
    """Plots SHAP summary plot (beeswarm) and optionally saves it."""
    plt.figure(figsize=(8, 6))
    
    # Handle binary class outputs where SHAP values might be a list or have 3 dimensions
    values_to_plot = shap_values
    if isinstance(shap_values, list):
        # KernelExplainer returns list for multi-class
        if len(shap_values) > 1:
            values_to_plot = shap_values[1]
    elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
        # Explainer returns Explanation object with shape (N, M, 2)
        values_to_plot = shap_values[:, :, 1]
        
    shap.summary_plot(values_to_plot, X, show=False)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_shap_local(shap_values, instance_idx: int, X, save_path=None):
    """Generates local explanation (waterfall) for a specific instance."""
    plt.figure(figsize=(8, 5))
    
    # Handle Explanation object vs raw array
    if hasattr(shap_values, "values"):
        explanation = shap_values[instance_idx]
        # Handle binary class dimension
        if len(explanation.values.shape) == 2:
            explanation = explanation[:, 1]
        shap.plots.waterfall(explanation, show=False)
    else:
        # Fallback if raw numpy array (like from KernelExplainer)
        values = shap_values
        if isinstance(shap_values, list):
            values = shap_values[1]
        shap.plots.force(
            base_value=0.5, # dummy base value
            shap_values=values[instance_idx],
            features=X.iloc[instance_idx],
            matplotlib=True,
            show=False
        )
        
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

