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
from configs.config import DATASET_PATH, PROCESSED_PATH, TARGET_COLUMN
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.utils.logger import setup_logger

logger = setup_logger("build_dataset", "logs/preprocessing.log")

def validate_dataset(df: pd.DataFrame, label: str):
    """Performs validation checks on the processed dataset."""
    logger.info(f"Validating {label} dataset...")
    
    # Check shape
    if df.shape[0] == 0:
        raise ValueError(f"Error: {label} dataset has 0 rows.")
    if df.shape[1] == 0:
        raise ValueError(f"Error: {label} dataset has 0 columns.")
        
    # Check for missing target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Error: Target column '{TARGET_COLUMN}' is missing in {label} dataset.")
        
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
    # Ensure processed directory exists
    processed_dir = Path(PROCESSED_PATH)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = Path(DATASET_PATH)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    logger.info(f"Loading raw dataset from {raw_path}")
    df_raw = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df_raw.shape}")

    # ==========================================
    # 1. Build Baseline Dataset
    # ==========================================
    logger.info("Building baseline dataset...")
    # Baseline reproduces early papers: basic features, minimal cleaning.
    # Typically keeps: sex, age, age_cat, race, priors_count, juv_fel_count, juv_misd_count, juv_other_count, c_charge_degree.
    baseline_features = [
        "sex", "age", "age_cat", "race", "priors_count",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
        "c_charge_degree", TARGET_COLUMN
    ]
    
    # Filter columns to only what's available
    baseline_features = [col for col in baseline_features if col in df_raw.columns]
    df_baseline = df_raw[baseline_features].copy()
    
    # Basic missing value fill
    df_baseline = df_baseline.dropna(subset=[TARGET_COLUMN])
    
    # Save raw baseline (unencoded categories) for EDA and dashboard
    df_baseline.to_csv(processed_dir / "compas_baseline_raw.csv", index=False)
    
    # Categorical encoding for model usage
    encoder_baseline = CategoricalEncoder(categorical_columns=["sex", "age_cat", "race", "c_charge_degree"])
    df_baseline_encoded = encoder_baseline.fit_transform(df_baseline)
    
    validate_dataset(df_baseline_encoded, "baseline")
    baseline_out_path = processed_dir / "compas_baseline.csv"
    df_baseline_encoded.to_csv(baseline_out_path, index=False)
    logger.info(f"Baseline dataset saved to {baseline_out_path} (shape: {df_baseline_encoded.shape})")

    # ==========================================
    # 2. Build Extended Dataset
    # ==========================================
    logger.info("Building extended dataset (leakage-aware)...")
    cleaner = DataCleaner()
    
    # Clean the dataset (filters standards, drops leakage/compas columns, fills NA)
    df_cleaned = cleaner.clean(df_raw, is_compas=True)
    
    # Keep raw categorical values for EDA/Explainability visualization
    df_cleaned.to_csv(processed_dir / "compas_extended_raw.csv", index=False)
    
    # Apply Categorical Encoder
    encoder_extended = CategoricalEncoder()
    df_extended_encoded = encoder_extended.fit_transform(df_cleaned)
    
    validate_dataset(df_extended_encoded, "extended")
    extended_out_path = processed_dir / "compas_extended.csv"
    df_extended_encoded.to_csv(extended_out_path, index=False)
    logger.info(f"Extended dataset saved to {extended_out_path} (shape: {df_extended_encoded.shape})")


if __name__ == "__main__":
    build_datasets()
