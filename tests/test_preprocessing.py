"""Unit tests for preprocessing module."""
import pandas as pd
import pytest
from src.preprocessing.feature_selector import remove_unwanted_columns

def test_remove_unwanted_columns():
    """Verifies that unwanted columns are dropped correctly."""
    data = {
        "id": [1, 2],
        "name": ["A", "B"],
        "age": [25, 30],
        "priors_count": [0, 3],
        "two_year_recid": [0, 1]
    }
    df = pd.DataFrame(data)
    df_filtered = remove_unwanted_columns(df)
    
    # 'id' and 'name' are identifiers defined in feature_config, so they should be removed
    assert "id" not in df_filtered.columns
    assert "name" not in df_filtered.columns
    assert "age" in df_filtered.columns
    assert "priors_count" in df_filtered.columns
