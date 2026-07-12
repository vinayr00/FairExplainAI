import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.models.model_factory import ModelFactory
from src.explainability.dice_explainer import generate_counterfactuals

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

    df = pd.read_csv(extended_path)
    df_raw = pd.read_csv(extended_raw_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    st.sidebar.title("DiCE Config")
    model_choice = st.sidebar.selectbox("Model to Interrogate", ["random_forest", "logistic_regression"])
    total_cfs = st.sidebar.slider("Number of counterfactuals", 1, 4, 2)

    st.subheader("Select Case Profile")
    case_idx = st.number_input("Case Index (from test set)", min_value=0, max_value=len(X_test)-1, value=0)

    # Show the raw profile
    st.markdown("**Original Case Profile:**")
    original_profile = df_raw.iloc[[X_test.index[case_idx]]]
    st.dataframe(original_profile, use_container_width=True)

    gen_btn = st.button("🔄 Generate Counterfactual Explanations")

    if gen_btn:
        with st.spinner("Generating counterfactuals..."):
            try:
                # Train model on the fly
                model = ModelFactory.get_model(model_choice, random_state=RANDOM_STATE)
                model.fit(X_train, y_train)

                # Query instance
                query_instance = X_test.iloc[[case_idx]]
                
                # Predict original label
                orig_pred = int(model.predict(query_instance)[0])
                if dataset == "compas":
                    pred_label = 'High Risk (1)' if orig_pred == 1 else 'Low Risk (0)'
                else:
                    pred_label = '>50K (1)' if orig_pred == 1 else '<=50K (0)'
                st.write(f"**Original Model Prediction**: `{pred_label}`")

                # Define continuous columns
                from src.preprocessing.feature_config import NUMERICAL_FEATURES
                continuous_cols = NUMERICAL_FEATURES
                
                # Call DiCE
                dice_exp = generate_counterfactuals(
                    model=model,
                    query_instances=query_instance,
                    data_interface=df_raw,
                    outcome_name=target_column,
                    continuous_features=continuous_cols,
                    total_CFs=total_cfs
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
