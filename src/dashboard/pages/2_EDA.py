import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from configs.config import PROCESSED_PATH

def main():
    st.set_page_config(page_title="EDA - FairExplainAI", layout="wide")
    
    from src.dashboard.components.sidebar import render_sidebar
    dataset = render_sidebar()
    
    from configs.config import DATASET_CONFIG
    target_column = DATASET_CONFIG[dataset]["target"]
    
    st.title("📈 Exploratory Data Analysis")
    st.markdown(f"""
    Explore the distributions, correlations, and potential biases in the processed **{dataset.upper()}** dataset.
    Visualizing these characteristics helps identify group disparities *before* training model classifiers.
    """)

    processed_dir = Path(PROCESSED_PATH)
    extended_raw_path = processed_dir / f"{dataset}_extended_raw.csv"

    if not extended_raw_path.exists():
        st.error(f"Processed dataset files not found for {dataset.upper()}. Please run the training pipeline first.")
        return

    df = pd.read_csv(extended_raw_path)

    # Set style for plots
    sns.set_theme(style="whitegrid")

    # Grid of distributions
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"⚖️ Target Distribution ({target_column})")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=target_column, palette="Set2", ax=ax)
        if dataset == "compas":
            ax.set_xticklabels(["Did Not Recidivate (0)", "Recidivated (1)"])
        else:
            ax.set_xticklabels(["<=50K (0)", ">50K (1)"])
        ax.set_xlabel("Target Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("👥 Race Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, y="race", order=df["race"].value_counts().index, palette="viridis", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Race")
        st.pyplot(fig)
        plt.close()

    # Disparity check: target by race & sex
    col3, col4 = st.columns(2)

    with col3:
        target_name = "Recidivism Rate" if dataset == "compas" else "High-Income Fraction (>50K)"
        st.subheader(f"📊 {target_name} by Race")
        recid_by_race = df.groupby("race")[target_column].mean().reset_index().sort_values(by=target_column)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=recid_by_race, x=target_column, y="race", palette="coolwarm", ax=ax)
        ax.set_xlabel(target_name)
        ax.set_ylabel("Race")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader(f"📊 {target_name} by Gender")
        recid_by_sex = df.groupby("sex")[target_column].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=recid_by_sex, x="sex", y=target_column, palette="Set1", ax=ax)
        ax.set_xlabel("Gender")
        ax.set_ylabel(target_name)
        st.pyplot(fig)
        plt.close()

    # Correlation Matrix
    st.subheader("🔗 Feature Correlation Heatmap")
    extended_encoded_path = processed_dir / f"{dataset}_extended.csv"
    if extended_encoded_path.exists():
        df_encoded = pd.read_csv(extended_encoded_path)
        corr = df_encoded.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run the training pipeline to generate the encoded feature set for correlation analysis.")

if __name__ == "__main__":
    main()
