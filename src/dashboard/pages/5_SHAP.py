import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.models.model_factory import ModelFactory
from src.explainability.shap_explainer import generate_shap_values
from src.dashboard.utils.caching import get_train_test_splits, get_cached_model, get_cached_shap_explainer_and_values

def main():
    st.set_page_config(page_title="SHAP Explanations - FairExplainAI", layout="wide")
    
    from src.dashboard.components.sidebar import render_sidebar
    dataset = render_sidebar()
    
    from configs.config import DATASET_CONFIG
    target_column = DATASET_CONFIG[dataset]["target"]
    
    st.title("🧩 SHAP Explainability")
    st.markdown(f"""
    Interpret model decisions globally and locally using **SHAP (SHapley Additive exPlanations)** on the **{dataset.upper()}** dataset.
    Understand which features push the model's prediction higher or lower.
    """)

    processed_dir = Path(PROCESSED_PATH)
    extended_path = processed_dir / f"{dataset}_extended.csv"

    if not extended_path.exists():
        st.error(f"Processed datasets not found for {dataset.upper()}. Run main.py first.")
        return

    X_train, X_test, y_train, y_test, _, _ = get_train_test_splits(dataset)

    st.sidebar.title("SHAP Config")
    model_choice = st.sidebar.selectbox("Model to Explain", ["random_forest", "xgboost", "logistic_regression"])

    st.subheader("Global Explanations: Feature Importance Summary")
    
    # Train/load model on the fly to get explainer
    try:
        model = get_cached_model(dataset, model_choice)

        # Plot beeswarm summary plot for a small sample of test instances for speed
        with st.spinner("Computing global SHAP values..."):
            shap_values = get_cached_shap_explainer_and_values(dataset, model_choice)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            values_to_plot = shap_values
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    values_to_plot = shap_values[1]
            elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                values_to_plot = shap_values[:, :, 1]
                
            shap.summary_plot(values_to_plot, X_test.iloc[:100], show=False)
            st.pyplot(fig)
            plt.close()

        st.subheader("Local Explanation: Single Case Interpretation")
        st.markdown("Select a case index from the test set to visualize its individual risk explanation:")
        
        # User input case index
        case_idx = st.number_input("Test Case Index", min_value=0, max_value=len(X_test)-1, value=0)
        
        # Display case features
        st.write("Instance Feature Profile:")
        st.dataframe(X_test.iloc[[case_idx]], use_container_width=True)

        with st.spinner("Generating waterfall explanation..."):
            single_shap = generate_shap_values(model, X_test.iloc[[case_idx]])
            
            fig, ax = plt.subplots(figsize=(8, 4))
            # Handle list vs Explanation objects
            if hasattr(single_shap, "values"):
                explanation = single_shap[0]
                if len(explanation.values.shape) == 2:
                    explanation = explanation[:, 1]
                shap.plots.waterfall(explanation, show=False)
            else:
                values = single_shap
                if isinstance(single_shap, list):
                    values = single_shap[1]
                shap.plots.force(
                    base_value=0.5,
                    shap_values=values[0],
                    features=X_test.iloc[case_idx],
                    matplotlib=True,
                    show=False
                )
                
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"Error computing SHAP: {e}")

if __name__ == "__main__":
    main()
