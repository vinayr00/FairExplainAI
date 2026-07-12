import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.models.model_factory import ModelFactory
from src.explainability.dice_explainer import generate_counterfactuals

def main():
    st.set_page_config(page_title="Counterfactuals - FairExplainAI", layout="wide")
    
    st.title("🔀 DiCE Counterfactual Explanations")
    st.markdown("""
    Generate **Counterfactual Explanations** using the DiCE (Diverse Counterfactual Explanations) library.
    Counterfactuals tell us: *What is the minimal set of changes (e.g. lowering priors count or increasing age) 
    that would flip the model's prediction from recidivating (high-risk) to not recidivating (low-risk)?*
    """)

    processed_dir = Path(PROCESSED_PATH)
    extended_path = processed_dir / "compas_extended.csv"
    extended_raw_path = processed_dir / "compas_extended_raw.csv"

    if not extended_path.exists() or not extended_raw_path.exists():
        st.error("Processed datasets not found. Run main.py first.")
        return

    df = pd.read_csv(extended_path)
    df_raw = pd.read_csv(extended_raw_path)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    st.sidebar.title("DiCE Config")
    model_choice = st.sidebar.selectbox("Model to Interrogate", ["random_forest", "logistic_regression"])
    total_cfs = st.sidebar.slider("Number of counterfactuals", 1, 4, 2)

    st.subheader("Select Offender Case Profile")
    case_idx = st.number_input("Case Index (from test set)", min_value=0, max_value=len(X_test)-1, value=0)

    # Show the raw profile
    st.markdown("**Original Offender Profile:**")
    original_profile = df_raw.iloc[[df_test_raw_idx := X_test.index[case_idx]]]
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
                st.write(f"**Original Model Prediction**: `{'High Risk (1)' if orig_pred == 1 else 'Low Risk (0)'}`")

                # Define continuous columns
                continuous_cols = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
                
                # Call DiCE
                dice_exp = generate_counterfactuals(
                    model=model,
                    query_instances=query_instance,
                    data_interface=df_raw,
                    outcome_name=TARGET_COLUMN,
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
