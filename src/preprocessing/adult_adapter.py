from pathlib import Path
import numpy as np
import pandas as pd
from configs.config import DATASET_CONFIG

class AdultAdapter:
    """Adapter for loading and cleaning the Adult Income dataset."""

    def __init__(self, filepath: str = None):
        if filepath is None:
            filepath = DATASET_CONFIG["adult"]["dataset_path"]
        self.filepath = Path(filepath)

    def get_dataframe(self) -> pd.DataFrame:
        """Loads, cleans, and merges the raw adult.data and adult.test partitions."""
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income"
        ]

        data_path = self.filepath / "adult.data"
        test_path = self.filepath / "adult.test"

        if not data_path.exists():
            raise FileNotFoundError(f"Adult data file not found at: {data_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Adult test file not found at: {test_path}")

        # Load adult.data (no header)
        df_train = pd.read_csv(data_path, header=None, names=columns, na_values=" ?")

        # Load adult.test (skip first comment line starting with '|1x3')
        df_test = pd.read_csv(
            test_path,
            header=None,
            names=columns,
            skiprows=1,
            na_values=" ?"
        )

        # Concatenate train and test sets
        df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        # Strip whitespaces from object columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()

        # Replace "?" or any placeholder NA values with np.nan
        df = df.replace("?", np.nan)

        # Normalize target column (income) by stripping the trailing dot from test records
        df["income"] = df["income"].str.replace(r"\.$", "", regex=True)

        # Map target classes to binary labels
        target_map = {"<=50K": 0, ">50K": 1}
        df["income"] = df["income"].map(target_map)

        # Drop rows where target is missing
        df = df.dropna(subset=["income"])
        df["income"] = df["income"].astype(int)

        return df
