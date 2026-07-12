import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.models.model_factory import ModelFactory
from src.fairness.threshold_optimizer import optimize_threshold
from src.fairness.metrics import get_fairness_summary, compute_fairness_metrics

def main():
    st.set_page_config(page_title="Fairness - FairExplainAI", layout="wide")
    
    from src.dashboard.components.sidebar import render_sidebar
    dataset = render_sidebar()
    
    from configs.config import DATASET_CONFIG
    target_column = DATASET_CONFIG[dataset]["target"]
    protected_attrs = DATASET_CONFIG[dataset]["protected"]
    
    st.title("⚖️ Bias Assessment & Mitigation")
    st.markdown("""
    Evaluate demographic biases and run fairness-aware mitigations. Use this section to run
    **Threshold Optimizer** post-processing to reduce performance gaps across sensitive groups (Race or Sex).
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
    df_train_raw, df_test_raw = train_test_split(
        df_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_raw[target_column]
    )

    # Mitigation configuration in UI
    st.sidebar.title("Fairness Settings")
    sensitive_attr = st.sidebar.selectbox("Sensitive Attribute", protected_attrs)
    constraint = st.sidebar.selectbox(
        "Mitigation Constraint", 
        ["demographic_parity", "equalized_odds", "true_positive_rate_parity"]
    )
    model_choice = st.sidebar.selectbox("Model to Mitigate", ["random_forest", "xgboost", "logistic_regression"])

    st.subheader(f"Analyzing Model: {model_choice.title()} on Protected Attribute: {sensitive_attr.title()}")

    # Training base model and applying mitigation
    with st.spinner("Applying fairness mitigation..."):
        try:
            # 1. Base Model
            base_model = ModelFactory.get_model(model_choice, random_state=RANDOM_STATE)
            base_model.fit(X_train, y_train)
            
            y_pred_base = base_model.predict(X_test)
            y_prob_base = base_model.predict_proba(X_test)[:, 1] if hasattr(base_model, "predict_proba") else None
            
            # Base metrics
            base_fair = get_fairness_summary(y_test, y_pred_base, df_test_raw[sensitive_attr])
            
            # 2. Mitigated Model (ThresholdOptimizer)
            # ThresholdOptimizer fits on train features and sensitive features
            opt = optimize_threshold(
                estimator=base_model,
                X=X_train,
                y=y_train,
                sensitive_features=df_train_raw[sensitive_attr],
                constraint=constraint
            )
            
            # Predict mitigated
            y_pred_mit = opt.predict(X_test, sensitive_features=df_test_raw[sensitive_attr])
            
            # Mitigated metrics
            mit_fair = get_fairness_summary(y_test, y_pred_mit, df_test_raw[sensitive_attr])
            
            # Show metrics side-by-side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Unmitigated Model")
                st.metric("Demographic Parity Diff", f"{base_fair['demographic_parity_difference']:.4f}")
                st.metric("Equalized Odds Diff", f"{base_fair['equalized_odds_difference']:.4f}")
                st.metric("Disparate Impact Ratio", f"{base_fair['disparate_impact_ratio']:.4f}")
                
            with col2:
                st.markdown("### Mitigated Model")
                st.metric("Demographic Parity Diff", f"{mit_fair['demographic_parity_difference']:.4f}", 
                          delta=f"{mit_fair['demographic_parity_difference'] - base_fair['demographic_parity_difference']:.4f}",
                          delta_color="inverse")
                st.metric("Equalized Odds Diff", f"{mit_fair['equalized_odds_difference']:.4f}",
                          delta=f"{mit_fair['equalized_odds_difference'] - base_fair['equalized_odds_difference']:.4f}",
                          delta_color="inverse")
                st.metric("Disparate Impact Ratio", f"{mit_fair['disparate_impact_ratio']:.4f}",
                          delta=f"{mit_fair['disparate_impact_ratio'] - base_fair['disparate_impact_ratio']:.4f}")

            # Group performance breakdown comparison
            st.subheader("Group-wise Selection and Error Rates")
            mf_base = compute_fairness_metrics(y_test, y_pred_base, df_test_raw[sensitive_attr])
            mf_mit = compute_fairness_metrics(y_test, y_pred_mit, df_test_raw[sensitive_attr])
            
            df_compare = pd.DataFrame({
                "Group": mf_base.by_group.index,
                "Base Selection Rate": mf_base.by_group["selection_rate"].values,
                "Mitigated Selection Rate": mf_mit.by_group["selection_rate"].values,
                "Base Accuracy": mf_base.by_group["accuracy"].values,
                "Mitigated Accuracy": mf_mit.by_group["accuracy"].values,
            })
            
            st.dataframe(df_compare, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to fit model or apply mitigation: {e}")

if __name__ == "__main__":
    main()
