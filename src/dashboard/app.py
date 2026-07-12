import streamlit as st
import json
from pathlib import Path
from configs.config import REPORTS_DIR, FIGURES_DIR

def main():
    st.set_page_config(
        page_title="FairExplainAI Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for rich aesthetics
    st.markdown("""
    <style>
        .title-text {
            font-family: 'Outfit', sans-serif;
            font-size: 3rem !important;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #2b5876, #4e4376);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .main-description {
            font-size: 1.15rem;
            line-height: 1.7;
            color: #4a4a4a;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="title-text">⚖️ FairExplainAI Framework</div>', unsafe_allow_html=True)
    
    from src.dashboard.components.sidebar import render_sidebar
    selected_dataset = render_sidebar()
    
    # Descriptions based on dataset
    if selected_dataset == "compas":
        dataset_description = "criminal recidivism predictions using the <strong>COMPAS dataset</strong>"
    else:
        dataset_description = "high income predictions (>50K) using the <strong>Adult Income dataset</strong>"

    st.markdown(f"""
    <p class="main-description">
    Welcome to the <strong>FairExplainAI</strong> interactive model analysis dashboard. 
    This framework is built to evaluate, mitigate, and explain machine learning biases in {dataset_description}. 
    Rather than introducing new algorithms, we reproduce the original Random Forest baseline and extend it through 
    leakage-aware preprocessing, multiple ML architectures, and post-processing fairness adjustments.
    </p>
    """, unsafe_allow_html=True)

    # Dynamic path resolution based on active selection
    from configs.config import BASE_DIR
    reports_dir = Path(BASE_DIR) / "reports" / selected_dataset
    summary_path = reports_dir / "summary_report.json"
    
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        st.subheader(f"📊 Latest Experimental Run Highlights ({selected_dataset.upper()})")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Baseline Accuracy vs. Extended Accuracy", 
                value=f"{summary['models'][1]['accuracy']:.2%}", 
                delta=f"{summary['models'][1]['accuracy'] - summary['models'][0]['accuracy']:.2%} (vs Baseline)"
            )
        with col2:
            st.metric(
                label="Demographic Parity Disparity", 
                value=f"{summary['models'][1]['demographic_parity_difference']:.4f}",
                delta=f"{summary['models'][1]['demographic_parity_difference'] - summary['models'][0]['demographic_parity_difference']:.4f} (vs Baseline)",
                delta_color="inverse"
            )
        with col3:
            rec = summary.get("recommendation", {})
            st.metric(
                label="Top Recommended Model",
                value=rec.get("model_name", "N/A"),
                delta=f"Utility Score: {rec.get('total_score', 0.0):.4f}"
            )
            
        st.info(f"**Policy Justification**: {rec.get('justification', 'No recommendation generated yet.')}")
    else:
        st.warning(f"⚠️ No summary report found for {selected_dataset.upper()}. Please run the training pipeline first using main.py (with DATASET='{selected_dataset}' in configs/config.py) to populate this dashboard.")
    st.subheader("📚 IEEE Research Objectives Covered")
    st.markdown("""
    - **Experiment A**: Baseline Paper Reproduction (Random Forest, raw features).
    - **Experiment B**: Leakage-Aware Feature Selection (removing decile scores and circular markers).
    - **Experiment C & G**: Multi-Model Classifiers & Hyperparameter Grid Search.
    - **Experiment D**: Fairness Mitigation via FairLearn post-processing and comparison.
    - **Experiment E**: SHAP Global and Local Explanations.
    - **Experiment F**: Counterfactual Explanations via DiCE.
    - **Experiment H**: Multi-Criteria recommendation engine selection.
    """)

    st.subheader("⚙️ System Architecture & Pipeline Flow")
    st.code("""
  [ Raw Dataset (COMPAS / Adult) ]
                 │
                 ▼
     [ Preprocessing & Adapter ]
                 │
                 ▼
  [ Baseline / Extended Features ]
                 │
                 ▼
     [ Model Training & Tuning ]
                 │
                 ▼
   [ Explainability & Fairness ] ──► [ SHAP & DiCE Counterfactuals ]
                 │
                 ▼
   [ Recommendation Engine ] ────► [ Multi-Criteria Ranking (AHP) ]
                 │
                 ▼
        [ Policy Decision ]
    """, language="text")

    col_cont, col_caps = st.columns(2)
    with col_cont:
        st.markdown("### 🔬 Research Contribution")
        st.markdown("""
        This framework extends the ethically-aware AI benchmark by proposing:
        - **Leakage-Aware Preprocessing**: Removing circular output predictors (e.g. COMPAS scores) to evaluate base demographics directly.
        - **Multi-Model Evaluation**: Testing tree structures (XGBoost, LightGBM, CatBoost) alongside traditional classifiers.
        - **Post-Processing Mitigation**: Applying `ThresholdOptimizer` constraints (Demographic Parity, Equal Opportunity) dynamically.
        """)
    with col_caps:
        st.markdown("### 🛠️ System Capabilities")
        st.markdown("""
        - **Cross-Dataset Performance**: Run benchmark evaluations on both the criminal justice **COMPAS** dataset and the socio-economic **Adult Income** dataset.
        - **Interactive Local Explainability**: Tweak query cases to visually trace decision changes using SHAP waterfalls and DiCE counterfactuals.
        - **Interactive Policy Query**: Adjust weights dynamically to score accuracy vs fairness trade-offs.
        """)
 
if __name__ == "__main__":
    main()

