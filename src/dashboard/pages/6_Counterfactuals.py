import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.models.model_factory import ModelFactory
from src.explainability.dice_explainer import generate_counterfactuals
from src.dashboard.utils.caching import get_train_test_splits, get_cached_model, get_cached_dice_explainer

def main():
    st.set_page_config(page_title="Counterfactuals - FairExplainAI", layout="wide")
    
    from src.dashboard.components.sidebar import render_sidebar
    dataset = render_sidebar()
    
    from configs.config import DATASET_CONFIG
    target_column = DATASET_CONFIG[dataset]["target"]
    
    st.title("🔀 DiCE Counterfactual Explanations")
    st.markdown(f"""
    Generate **Counterfactual Explanations** using the DiCE (Diverse Counterfactual Explanations) library on the **{dataset.upper()}** dataset.
    Counterfactuals tell us: *What is the minimal set of changes (e.g. changing numeric or categorical parameters) 
    that would flip the model's prediction?*
    """)

    processed_dir = Path(PROCESSED_PATH)
    extended_path = processed_dir / f"{dataset}_extended.csv"
    extended_raw_path = processed_dir / f"{dataset}_extended_raw.csv"

    if not extended_path.exists() or not extended_raw_path.exists():
        st.error(f"Processed datasets not found for {dataset.upper()}. Run main.py first.")
        return

    X_train, X_test, y_train, y_test, df_train_raw, df_test_raw = get_train_test_splits(dataset)

    st.sidebar.title("DiCE Config")
    model_choice = st.sidebar.selectbox("Model to Interrogate", ["random_forest", "logistic_regression"])
    total_cfs = st.sidebar.slider("Number of counterfactuals", 1, 4, 2)

    st.subheader("Select Case Profile")
    case_idx = st.number_input("Case Index (from test set)", min_value=0, max_value=len(X_test)-1, value=0)

    # Show the raw profile
    st.markdown("**Original Case Profile:**")
    original_profile = df_test_raw.iloc[[case_idx]]
    st.dataframe(original_profile, use_container_width=True)

    gen_btn = st.button("🔄 Generate Counterfactual Explanations")

    if gen_btn:
        with st.spinner("Generating counterfactuals..."):
            try:
                # Retrieve cached model and input instance
                model = get_cached_model(dataset, model_choice)
                query_instance = X_test.iloc[[case_idx]]
                
                # Predict original label and compute display categories
                orig_pred = int(model.predict(query_instance)[0])
                if dataset == "compas":
                    pred_label = 'High Risk (1)' if orig_pred == 1 else 'Low Risk (0)'
                    cf_label = 'Low Risk (0)' if orig_pred == 1 else 'High Risk (1)'
                else:
                    pred_label = '>50K (1)' if orig_pred == 1 else '<=50K (0)'
                    cf_label = '<=50K (0)' if orig_pred == 1 else '>50K (1)'
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🔍 Original Model Prediction")
                    if orig_pred == 1:
                        st.error(f"**{pred_label}**")
                    else:
                        st.success(f"**{pred_label}**")
                with col2:
                    st.markdown("##### 🔄 Target Flipped Prediction")
                    if orig_pred == 1:
                        st.success(f"**{cf_label}**")
                    else:
                        st.error(f"**{cf_label}**")
                
                # Retrieve cached DiCE explainer object and generate live counterfactuals
                exp = get_cached_dice_explainer(dataset, model_choice)
                
                dice_exp = exp.generate_counterfactuals(
                    query_instance, total_CFs=total_cfs, desired_class="opposite"
                )
                
                if dice_exp:
                    st.markdown("**Diverse Counterfactual Examples (Flipped prediction):**")
                    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                    st.dataframe(cf_df, use_container_width=True)
                    st.info("The columns above highlight how the features could change to output the opposite risk category.")
                else:
                    st.warning("Could not generate counterfactuals for this profile.")
                    
            except Exception as e:
                st.error(f"Error generating counterfactuals: {e}")

if __name__ == "__main__":
    main()
