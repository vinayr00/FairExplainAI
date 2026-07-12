import streamlit as st
import pandas as pd
from pathlib import Path
from configs.config import PROCESSED_PATH, TARGET_COLUMN, PROTECTED_ATTRIBUTES

def main():
    st.set_page_config(page_title="Dataset Overview - FairExplainAI", layout="wide")
    
    st.title("📊 Dataset Overview")
    st.markdown("""
    This page provides an overview of the **COMPAS Dataset** and explains the difference between the **Baseline** feature set
    (which reproduces the original research) and the **Extended** feature set (which utilizes leakage-aware feature selection).
    """)

    processed_dir = Path(PROCESSED_PATH)
    baseline_raw_path = processed_dir / "compas_baseline_raw.csv"
    extended_raw_path = processed_dir / "compas_extended_raw.csv"

    if not baseline_raw_path.exists() or not extended_raw_path.exists():
        st.error("Processed dataset files not found. Please run the training pipeline first using `main.py` or through the Model Training page.")
        return

    df_base = pd.read_csv(baseline_raw_path)
    df_ext = pd.read_csv(extended_raw_path)

    # High-level Metrics Card Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Raw Dataset Records", value="7,214", delta=None)
    with col2:
        st.metric(label="Baseline Dataset Shape", value=f"{df_base.shape[0]} × {df_base.shape[1]}", delta=None)
    with col3:
        st.metric(label="Extended Dataset Shape", value=f"{df_ext.shape[0]} × {df_ext.shape[1]}", delta="-1,042 rows (cleaned)")

    st.subheader("Scientific Justifications for Preprocessing & Cleaning")
    
    st.markdown("""
    To ensure a rigorous, publication-grade experimental design, the preprocessing pipeline implements ProPublica's filtering rules 
    and applies a leakage-aware feature selection approach:
    
    1. **Days Between Screening and Arrest**: We remove records where the number of days is greater than 30 or less than -30, ensuring that the recidivism prediction is tied to the correct underlying offense.
    2. **Recidivism Target Quality**: We drop records where `is_recid` is -1, indicating invalid recidivism tracking.
    3. **Charge Severity Filtering**: Minor traffic violations (labeled as degree 'O') are dropped as they do not constitute criminal recidivism.
    4. **Leakage & Circularity Prevention**: We explicitly remove the COMPAS output variables (such as `decile_score`, `score_text`, `v_decile_score`). Including these columns would result in a target leakage model that merely replicates the proprietary COMPAS black-box, rather than learning directly from original offender features.
    """)

    # Visualizing baseline vs extended dataframes
    tab1, tab2 = st.tabs(["Baseline Feature Set (Reproduced)", "Extended Feature Set (Leakage-Aware)"])
    
    with tab1:
        st.markdown("**Baseline Dataset Sample** (showing first 10 rows):")
        st.dataframe(df_base.head(10), use_container_width=True)
        st.info(f"Target Column: `{TARGET_COLUMN}` | Protected Attributes: `{', '.join(PROTECTED_ATTRIBUTES)}`")

    with tab2:
        st.markdown("**Extended Dataset Sample** (showing first 10 rows):")
        st.dataframe(df_ext.head(10), use_container_width=True)
        st.info("Notice that all COMPAS internal scores, descriptions, violence-only indicators, and administrative columns have been safely removed to prevent leakage.")

if __name__ == "__main__":
    main()
