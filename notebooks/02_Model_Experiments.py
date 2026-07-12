"""
Model Experiments & Hyperparameter Tuning
-------------------------------------------
This script trains multiple machine learning classifiers on the processed
COMPAS dataset, performs GridSearchCV hyperparameter tuning, and evaluates
the performance of the models on the test set.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path to allow imports when running directly
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from configs.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, PROCESSED_PATH
from src.models.model_factory import ModelFactory
from src.models.hyperparameter_tuning import tune_model
from src.evaluation.metrics import evaluate_predictions

def run_experiments():
    data_path = root_dir / PROCESSED_PATH / "compas_extended.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}. Please run build_dataset first.")
        return

    print(f"Loading extended processed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    models_list = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    results = []

    for model_name in models_list:
        print(f"\n--- Tuning Hyperparameters for: {model_name} ---")
        try:
            base_model = ModelFactory.get_model(model_name, random_state=RANDOM_STATE)
            best_estimator, best_params = tune_model(
                model_name=model_name,
                base_estimator=base_model,
                X=X_train,
                y=y_train,
                search_type="grid",
                cv=3
            )
            
            # Evaluate on test set
            y_pred = best_estimator.predict(X_test)
            y_prob = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, "predict_proba") else None
            perf = evaluate_predictions(y_test, y_pred, y_prob)
            
            results.append({
                "model_name": model_name,
                "best_params": best_params,
                **perf
            })
            
            print(f"Test Accuracy: {perf['accuracy']:.4f}")
            print(f"Best Params: {best_params}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    df_results = pd.DataFrame(results)
    print("\n=========================================")
    print("Final Model Experiments Summary:")
    print("=========================================")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    run_experiments()
