"""FileManager for loading/saving data objects, models, and json files."""
import json
import pickle
from pathlib import Path
import pandas as pd

class FileManager:
    """Utility class to read/write objects to the filesystem."""

    @staticmethod
    def save_pickle(obj, filepath):
        """Pickles and saves python objects (like model estimators)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(filepath):
        """Loads and returns pickled python objects."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_json(data: dict, filepath):
        """Saves dictionary to JSON format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(filepath) -> dict:
        """Loads and returns JSON dictionary."""
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def save_csv(df: pd.DataFrame, filepath, index=False):
        """Saves Pandas DataFrame to CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=index)

    @staticmethod
    def save_text(text: str, filepath):
        """Saves plain text to a file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

