import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from configs.config import RESULTS_DIR, FIGURES_DIR
from src.evaluation.recommendation_engine import RecommendationEngine

def main():
    st.set_page_config(page_title="Model Comparison & Recommendation - FairExplainAI", layout="wide")
    
    st.title("🏆 Model Comparison & Recommendation")
    st.markdown("""
    Compare the baseline and all extended models across multiple criteria (Accuracy, Fairness, Efficiency). 
    Adjust the sliding priorities to query the **Recommendation Engine** for the best overall classifier.
    """)

    results_dir = Path(RESULTS_DIR)
    comparison_csv = results_dir / "model_comparison.csv"

    if not comparison_csv.exists():
        st.error("Model comparison data not found. Please run the training pipeline first using main.py.")
        st.info("You can also run the training process by triggering main.py via the dashboard.")
        return

    # Load comparison records
    df = pd.read_csv(comparison_csv)

    # Sidebar parameters: User priority weights
    st.sidebar.title("Recommendation Weights")
    st.sidebar.markdown("Adjust priorities to dynamically rank the models (must sum to 1.0):")
    
    acc_w = st.sidebar.slider("Performance (Accuracy/F1)", 0.0, 1.0, 0.40, 0.05)
    fair_w = st.sidebar.slider("Fairness (DP / Equalized Odds)", 0.0, 1.0, 0.40, 0.05)
    eff_w = st.sidebar.slider("Efficiency (Train/Inference Time)", 0.0, 1.0, 0.20, 0.05)

    # Check weights sum
    total_w = acc_w + fair_w + eff_w
    if abs(total_w - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total_w:.2f}. Please adjust them to sum to 1.0.")
    
    # Live Recommendation query
    st.subheader("Interactive Model Ranking")
    
    models_metrics = df[[
        "model_name", "accuracy", "f1_score", "roc_auc", 
        "demographic_parity_difference", "equalized_odds_difference",
        "training_time", "inference_time"
    ]].to_dict(orient="records")

    # Run recommendation
    engine = RecommendationEngine(
        priorities={
            "accuracy_weight": acc_w,
            "fairness_weight": fair_w,
            "efficiency_weight": eff_w
        }
    )
    
    df_ranked = engine.rank_models(models_metrics)
    best_rec = engine.recommend_best(models_metrics)

    # Show top recommendation
    if best_rec:
        st.success(f"🏆 **Top Recommended Model**: **{best_rec['model_name']}** (Utility Score: {best_rec['total_score']:.4f})")
        st.info(f"**Justification**: {best_rec['justification']}")

    # Render comparisons table
    st.markdown("### Performance & Fairness Metrics comparison")
    
    # Style DataFrame for better display
    styled_df = df_ranked[[
        "rank", "model_name", "total_score", "accuracy", "f1_score", "roc_auc",
        "demographic_parity_difference", "equalized_odds_difference", "training_time"
    ]].copy()
    
    styled_df.columns = [
        "Rank", "Model Name", "Utility Score", "Accuracy", "F1 Score", "ROC-AUC",
        "Demographic Parity Diff", "Equalized Odds Diff", "Train Time (s)"
    ]
    
    st.dataframe(styled_df.style.format({
        "Utility Score": "{:.4f}",
        "Accuracy": "{:.2%}",
        "F1 Score": "{:.2%}",
        "ROC-AUC": "{:.2%}",
        "Demographic Parity Diff": "{:.4f}",
        "Equalized Odds Diff": "{:.4f}",
        "Train Time (s)": "{:.4f}"
    }), use_container_width=True)

    # Plot Tradeoff (Accuracy vs Demographic Parity Difference)
    st.subheader("📉 Fairness-Accuracy Pareto Frontier")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.scatterplot(
        data=df_ranked,
        x="accuracy",
        y="demographic_parity_difference",
        hue="model_name",
        s=120,
        ax=ax,
        palette="viridis"
    )
    
    # Annotate points
    for idx, row in df_ranked.iterrows():
        ax.annotate(
            row["model_name"], 
            (row["accuracy"], row["demographic_parity_difference"]),
            textcoords="offset points", 
            xytext=(5, 5), 
            ha="left",
            fontsize=9
        )
        
    ax.set_xlabel("Accuracy (Higher is better)")
    ax.set_ylabel("Demographic Parity Difference (Lower is better)")
    ax.set_title("Model Tradeoff: Accuracy vs demographic disparity")
    st.pyplot(fig)
    plt.close()

if __name__ == "__main__":
    main()
