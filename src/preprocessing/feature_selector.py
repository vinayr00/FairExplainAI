"""
Feature Selector
Removes unwanted columns based on
feature configuration.
"""

import pandas as pd

from .feature_config import (
    IDENTIFIER_COLUMNS,
    DATE_COLUMNS,
    LEAKAGE_COLUMNS,
    COMPAS_COLUMNS,
    VIOLENCE_COLUMNS,
    DUPLICATE_COLUMNS
)

def remove_unwanted_columns(df: pd.DataFrame):

    columns_to_drop = (
        IDENTIFIER_COLUMNS
        + DATE_COLUMNS
        + LEAKAGE_COLUMNS
        + COMPAS_COLUMNS
        + VIOLENCE_COLUMNS
        + DUPLICATE_COLUMNS
    )

    columns_to_drop = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    return df.drop(columns=columns_to_drop)