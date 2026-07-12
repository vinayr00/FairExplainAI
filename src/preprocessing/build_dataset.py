"""
build_dataset.py
----------------
This script runs the preprocessing pipeline to generate:
1. `compas_baseline.csv`: Minimal cleaning, reproducing the baseline feature set.
2. `compas_extended.csv`: Strict ProPublica filtering and leakage-aware feature selection.
"""

import logging
from pathlib import Path
import pandas as pd
from configs.config import PROCESSED_PATH
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.utils.logger import setup_logger

logger = setup_logger("build_dataset", "logs/preprocessing.log")

def validate_dataset(df: pd.DataFrame, label: str):
    """Performs validation checks on the processed dataset."""
    import configs.config as cfg
    logger.info(f"Validating {label} dataset...")
    
    # Check shape
    if df.shape[0] == 0:
        raise ValueError(f"Error: {label} dataset has 0 rows.")
    if df.shape[1] == 0:
        raise ValueError(f"Error: {label} dataset has 0 columns.")
        
    # Check for missing target column
    if cfg.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Error: Target column '{cfg.TARGET_COLUMN}' is missing in {label} dataset.")
        
    # Check for missing values in critical columns
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Warning: {label} dataset contains missing values:\n{null_counts[null_counts > 0]}")
    else:
        logger.info(f"Success: No missing values in {label} dataset.")

    # Check for duplicate records
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Warning: {label} dataset contains {dup_count} duplicate rows.")
    else:
        logger.info(f"Success: No duplicate records in {label} dataset.")


def build_datasets():
    """Generates and saves the baseline and extended datasets."""
    from configs.config import DATASET, TARGET_COLUMN
    from src.preprocessing.dataset_loader import get_dataset_adapter
    from src.preprocessing.feature_config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    
    # Ensure processed directory exists
    processed_dir = Path(PROCESSED_PATH)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Retrieve configured adapter and raw dataframe
    adapter = get_dataset_adapter()
    logger.info(f"Loading raw dataset using adapter for: {DATASET}")
    df_raw = adapter.get_dataframe()
    logger.info(f"Raw shape: {df_raw.shape}")

    # ==========================================
    # 1. Build Baseline Dataset
    # ==========================================
    logger.info(f"Building baseline dataset for {DATASET}...")
    baseline_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
    
    # Filter columns to only what's available
    baseline_features = [col for col in baseline_features if col in df_raw.columns]
    df_baseline = df_raw[baseline_features].copy()
    
    # Basic missing value fill
    df_baseline = df_baseline.dropna(subset=[TARGET_COLUMN])
    
    # Save raw baseline (unencoded categories) for EDA and dashboard
    df_baseline.to_csv(processed_dir / f"{DATASET}_baseline_raw.csv", index=False)
    
    # Categorical encoding for model usage
    encoder_baseline = CategoricalEncoder(categorical_columns=CATEGORICAL_FEATURES)
    df_baseline_encoded = encoder_baseline.fit_transform(df_baseline)
    
    validate_dataset(df_baseline_encoded, f"{DATASET}_baseline")
    baseline_out_path = processed_dir / f"{DATASET}_baseline.csv"
    df_baseline_encoded.to_csv(baseline_out_path, index=False)
    logger.info(f"Baseline dataset saved to {baseline_out_path} (shape: {df_baseline_encoded.shape})")

    # ==========================================
    # 2. Build Extended Dataset
    # ==========================================
    logger.info(f"Building extended dataset (leakage-aware) for {DATASET}...")
    cleaner = DataCleaner()
    
    # Clean the dataset (filters standards for COMPAS, drops leakage columns, fills NA)
    df_cleaned = cleaner.clean(df_raw, is_compas=(DATASET == "compas"))
    
    # Keep raw categorical values for EDA/Explainability visualization
    df_cleaned.to_csv(processed_dir / f"{DATASET}_extended_raw.csv", index=False)
    
    # Apply Categorical Encoder
    encoder_extended = CategoricalEncoder(categorical_columns=CATEGORICAL_FEATURES)
    df_extended_encoded = encoder_extended.fit_transform(df_cleaned)
    
    validate_dataset(df_extended_encoded, f"{DATASET}_extended")
    extended_out_path = processed_dir / f"{DATASET}_extended.csv"
    df_extended_encoded.to_csv(extended_out_path, index=False)
    logger.info(f"Extended dataset saved to {extended_out_path} (shape: {df_extended_encoded.shape})")


if __name__ == "__main__":
    build_datasets()
