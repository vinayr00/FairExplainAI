import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from configs.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, DATASET_CONFIG
from src.models.model_factory import ModelFactory
from src.explainability.shap_explainer import generate_shap_values
import dice_ml

@st.cache_data
def load_dataset_raw(dataset: str) -> pd.DataFrame:
    """Loads raw categorical dataset (e.g. compas_extended_raw.csv)."""
    processed_dir = Path(PROCESSED_PATH)
    extended_raw_path = processed_dir / f"{dataset}_extended_raw.csv"
    return pd.read_csv(extended_raw_path)

@st.cache_data
def load_dataset_encoded(dataset: str) -> pd.DataFrame:
    """Loads encoded numerical dataset (e.g. compas_extended.csv)."""
    processed_dir = Path(PROCESSED_PATH)
    extended_path = processed_dir / f"{dataset}_extended.csv"
    return pd.read_csv(extended_path)

@st.cache_data
def get_train_test_splits(dataset: str):
    """Generates train/test splits for both encoded and raw data, keyed by dataset name."""
    df_encoded = load_dataset_encoded(dataset)
    df_raw = load_dataset_raw(dataset)
    
    target_column = DATASET_CONFIG[dataset]["target"]
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    df_train_raw, df_test_raw = train_test_split(
        df_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_raw[target_column]
    )
    
    return X_train, X_test, y_train, y_test, df_train_raw, df_test_raw

@st.cache_resource
def get_cached_model(dataset: str, model_choice: str):
    """Trains and caches the specified classifier with default parameters."""
    X_train, _, y_train, _, _, _ = get_train_test_splits(dataset)
    
    model = ModelFactory.get_model(model_choice, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def get_cached_shap_explainer_and_values(dataset: str, model_choice: str):
    """Computes global SHAP values for the first 100 test instances."""
    model = get_cached_model(dataset, model_choice)
    _, X_test, _, _, _, _ = get_train_test_splits(dataset)
    
    shap_values = generate_shap_values(model, X_test.iloc[:100])
    return shap_values

@st.cache_resource
def get_cached_dice_explainer(dataset: str, model_choice: str):
    """Initializes and caches the DiCE explainer object for live counterfactual queries."""
    model = get_cached_model(dataset, model_choice)
    df_encoded = load_dataset_encoded(dataset)
    target_column = DATASET_CONFIG[dataset]["target"]
    
    from src.preprocessing.feature_config import NUMERICAL_FEATURES
    continuous_features = NUMERICAL_FEATURES
    
    # Identify numeric categorical columns that need casting to prevent LossySetitemError in Pandas 2.x
    cols_to_cast = {}
    for col in df_encoded.columns:
        if col != target_column and col not in continuous_features:
            if pd.api.types.is_numeric_dtype(df_encoded[col]):
                cols_to_cast[col] = df_encoded[col].dtype
                
    # Wrap model to cast inputs back to training types if casting is needed
    if cols_to_cast:
        class ModelWrapper:
            def __init__(self, inner_model, cols_to_cast):
                self.inner_model = inner_model
                self.cols_to_cast = cols_to_cast
                
            def _prepare_input(self, X):
                X_copy = X.copy()
                for col, dtype in self.cols_to_cast.items():
                    if col in X_copy.columns:
                        X_copy[col] = X_copy[col].astype(dtype)
                return X_copy
                
            def predict(self, X, **kwargs):
                return self.inner_model.predict(self._prepare_input(X), **kwargs)
                
            def predict_proba(self, X, **kwargs):
                return self.inner_model.predict_proba(self._prepare_input(X), **kwargs)
                
            def __getattr__(self, name):
                return getattr(self.inner_model, name)
                
        model = ModelWrapper(model, cols_to_cast)
        
    m = dice_ml.Model(model=model, backend="sklearn")
    d = dice_ml.Data(
        dataframe=df_encoded,
        continuous_features=continuous_features,
        outcome_name=target_column,
    )
    
    exp = dice_ml.Dice(d, m, method="random")
    return exp
