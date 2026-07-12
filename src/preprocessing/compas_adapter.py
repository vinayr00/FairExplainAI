import pandas as pd
from configs.config import DATASET_CONFIG

class CompasAdapter:
    """Adapter for the COMPAS dataset loading logic."""

    def __init__(self, filepath: str = None):
        if filepath is None:
            # Fall back to dataset_path in configs
            filepath = DATASET_CONFIG["compas"]["dataset_path"]
        self.filepath = filepath

    def get_dataframe(self) -> pd.DataFrame:
        """Loads and returns the raw COMPAS dataset."""
        return pd.read_csv(self.filepath)
