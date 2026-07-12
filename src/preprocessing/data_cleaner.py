import pandas as pd
from typing import List, Optional
from src.preprocessing.feature_config import COLUMNS_TO_DROP

class DataCleaner:
    """Class to handle data cleaning and filtering, specifically optimized for the COMPAS dataset."""

    def __init__(self, columns_to_drop: Optional[List[str]] = None):
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else COLUMNS_TO_DROP

    def filter_compas_standards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies ProPublica's standard filtering criteria for the COMPAS dataset:
        
        1. Remove rows where days_b_screening_arrest is > 30 or < -30.
        2. Remove rows where is_recid is -1 (no recidivism information).
        3. Remove rows where c_charge_degree is 'O' (ordinary/traffic offenses).
        4. Remove rows where score_text is N/A.
        """
        df_clean = df.copy()

        # 1. Filter by days_b_screening_arrest
        if "days_b_screening_arrest" in df_clean.columns:
            df_clean = df_clean[
                (df_clean["days_b_screening_arrest"] <= 30) & 
                (df_clean["days_b_screening_arrest"] >= -30)
            ]

        # 2. Filter out invalid recidivism records
        if "is_recid" in df_clean.columns:
            df_clean = df_clean[df_clean["is_recid"] != -1]

        # 3. Filter out charge degree 'O' (not felony or misdemeanor)
        if "c_charge_degree" in df_clean.columns:
            df_clean = df_clean[df_clean["c_charge_degree"] != "O"]

        # 4. Filter out rows with missing score_text
        if "score_text" in df_clean.columns:
            df_clean = df_clean.dropna(subset=["score_text"])

        return df_clean

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unneeded columns from the DataFrame."""
        existing_drops = [col for col in self.columns_to_drop if col in df.columns]
        return df.drop(columns=existing_drops)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles any remaining missing values in critical feature columns."""
        from configs.config import TARGET_COLUMN, PROTECTED_ATTRIBUTES
        from src.preprocessing.feature_config import NUMERICAL_FEATURES

        df_clean = df.copy()
        
        # Drop rows where target or protected attributes are missing
        critical_cols = [TARGET_COLUMN] + PROTECTED_ATTRIBUTES
        existing_critical = [col for col in critical_cols if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=existing_critical)
        
        # Fill numerical features with median
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        return df_clean

    def clean(self, df: pd.DataFrame, is_compas: bool = None) -> pd.DataFrame:
        """Executes the full pipeline of data cleaning steps."""
        from configs.config import DATASET
        if is_compas is None:
            is_compas = (DATASET == "compas")
            
        if is_compas:
            df = self.filter_compas_standards(df)
        df = self.handle_missing_values(df)
        df = self.drop_columns(df)
        return df
