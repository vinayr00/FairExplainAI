import streamlit as st
import pandas as pd
from pathlib import Path
from configs.config import PROCESSED_PATH, TARGET_COLUMN, PROTECTED_ATTRIBUTES

def main():
    st.set_page_config(page_title="Dataset Overview - FairExplainAI", layout="wide")
    
    from src.dashboard.components.sidebar import render_sidebar
    dataset = render_sidebar()
    
    from configs.config import DATASET_CONFIG
    target_column = DATASET_CONFIG[dataset]["target"]
    protected_attributes = DATASET_CONFIG[dataset]["protected"]
    
    st.title("📊 Dataset Overview")
    st.markdown(f"""
    This page provides an overview of the **{dataset.upper()} Dataset** and explains the difference between the **Baseline** feature set
    (which reproduces the original research) and the **Extended** feature set (which utilizes leakage-aware feature selection).
    """)

    processed_dir = Path(PROCESSED_PATH)
    baseline_raw_path = processed_dir / f"{dataset}_baseline_raw.csv"
    extended_raw_path = processed_dir / f"{dataset}_extended_raw.csv"

    if not baseline_raw_path.exists() or not extended_raw_path.exists():
        st.error(f"Processed dataset files not found for {dataset.upper()}. Please run the training pipeline first using `main.py` or through the Model Training page.")
        return

    df_base = pd.read_csv(baseline_raw_path)
    df_ext = pd.read_csv(extended_raw_path)

    # High-level Metrics Card Row
    col1, col2, col3 = st.columns(3)
    with col1:
        raw_count = "7,214" if dataset == "compas" else "48,842"
        st.metric(label="Raw Dataset Records", value=raw_count, delta=None)
    with col2:
        st.metric(label="Baseline Dataset Shape", value=f"{df_base.shape[0]} × {df_base.shape[1]}", delta=None)
    with col3:
        rows_delta = "-1,042 rows (cleaned)" if dataset == "compas" else f"-{df_base.shape[0] - df_ext.shape[0]} rows (cleaned)"
        st.metric(label="Extended Dataset Shape", value=f"{df_ext.shape[0]} × {df_ext.shape[1]}", delta=rows_delta)

    st.subheader("Scientific Justifications for Preprocessing & Cleaning")
    
    if dataset == "compas":
        st.markdown("""
        To ensure a rigorous, publication-grade experimental design, the preprocessing pipeline implements ProPublica's filtering rules 
        and applies a leakage-aware feature selection approach:
        
        1. **Days Between Screening and Arrest**: We remove records where the number of days is greater than 30 or less than -30, ensuring that the recidivism prediction is tied to the correct underlying offense.
        2. **Recidivism Target Quality**: We drop records where `is_recid` is -1, indicating invalid recidivism tracking.
        3. **Charge Severity Filtering**: Minor traffic violations (labeled as degree 'O') are dropped as they do not constitute criminal recidivism.
        4. **Leakage & Circularity Prevention**: We explicitly remove the COMPAS output variables (such as `decile_score`, `score_text`, `v_decile_score`). Including these columns would result in a target leakage model that merely replicates the proprietary COMPAS black-box, rather than learning directly from original offender features.
        """)
    else:
        st.markdown("""
        To ensure a rigorous evaluation on the Adult Income dataset, the preprocessing and cleaning pipeline handles:
        
        1. **Whitespace and Format Normalization**: Leading and trailing whitespaces are stripped from all categorical columns, and standard headers are applied.
        2. **Missing Value Handling**: Missing indicator values represented as `"?"` are standardized to standard null values (`np.nan`) to enable unified preprocessing imputation.
        3. **Label Alignment**: Trailing dots are stripped from the `income` label (as present in the original `adult.test` partition), and target classes are mapped to binary integers (`0` for `<=50K` and `1` for `>50K`).
        """)

    # Visualizing baseline vs extended dataframes
    tab1, tab2 = st.tabs(["Baseline Feature Set", "Extended Feature Set"])
    
    with tab1:
        st.markdown(f"**Baseline Dataset Sample** (showing first 10 rows):")
        st.dataframe(df_base.head(10), use_container_width=True)
        st.info(f"Target Column: `{target_column}` | Protected Attributes: `{', '.join(protected_attributes)}`")

    with tab2:
        st.markdown(f"**Extended Dataset Sample** (showing first 10 rows):")
        st.dataframe(df_ext.head(10), use_container_width=True)
        st.info(f"Notice that administrative columns and unwanted features have been handled/removed in the extended dataset.")

if __name__ == "__main__":
    main()
