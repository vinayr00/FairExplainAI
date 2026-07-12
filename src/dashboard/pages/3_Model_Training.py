import streamlit as st
import pandas as pd
import time
import tracemalloc
from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.plots import plot_confusion_matrix

def main():
    st.set_page_config(page_title="Model Training - FairExplainAI", layout="wide")
    
    st.title("⚙️ Model Training & Experiments")
    st.markdown("""
    Train and customize machine learning classifiers on the processed COMPAS datasets. 
    Select a model, adjust hyperparameters, and instantly review test set performance.
    """)

    processed_dir = Path(PROCESSED_PATH)
    extended_path = processed_dir / "compas_extended.csv"

    if not extended_path.exists():
        st.error("Processed datasets not found. Run main.py first.")
        return

    # Load data on-the-fly for interactive training
    df = pd.read_csv(extended_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Sidebar parameters
    st.sidebar.title("Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Choose Classifier",
        ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
    )

    # Dynamic inputs based on model choice
    params = {}
    if model_choice == "logistic_regression":
        params["C"] = st.sidebar.slider("Regularization C", 0.01, 10.0, 1.0)
        params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 2000, 1000)
    elif model_choice == "decision_tree":
        params["max_depth"] = st.sidebar.slider("Max Depth", 2, 20, 5)
        params["criterion"] = st.sidebar.selectbox("Split Criterion", ["gini", "entropy"])
    elif model_choice == "random_forest":
        params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 300, 100)
        params["max_depth"] = st.sidebar.slider("Max Depth", 2, 20, 8)
    elif model_choice in ("xgboost", "lightgbm", "catboost"):
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 300, 100)
        params["max_depth"] = st.sidebar.slider("Max Depth", 2, 10, 5)
        params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)

    train_btn = st.sidebar.button("🚀 Train Model")

    if train_btn:
        st.info(f"Training {model_choice} with parameters: {params}")
        
        # Track memory and runtime
        tracemalloc.start()
        start_time = time.time()
        
        try:
            model = ModelFactory.get_model(model_choice, random_state=RANDOM_STATE, **params)
            model.fit(X_train, y_train)
            
            elapsed_time = time.time() - start_time
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            st.success(f"Model trained successfully in {elapsed_time:.4f} seconds!")
            
            # Predict
            start_infer = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - start_infer
            
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Performance metrics
            metrics = evaluate_predictions(y_test, y_pred, y_prob)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{metrics['accuracy']:.4%}")
            with col2:
                st.metric("Test F1-Score", f"{metrics['f1_score']:.4%}")
            with col3:
                st.metric("Test ROC-AUC", f"{metrics['roc_auc']:.4%}" if "roc_auc" in metrics else "N/A")
            with col4:
                st.metric("Peak Memory Used", f"{peak_mem / (1024 * 1024):.3f} MB")

            # Plot confusion matrix
            st.subheader("Confusion Matrix")
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error during training: {e}")
            tracemalloc.stop()
    else:
        st.info("Adjust settings in the sidebar and click **Train Model** to run training.")

if __name__ == "__main__":
    main()
