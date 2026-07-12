import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder

class CategoricalEncoder:
    """Class to handle categorical variable encoding using one-hot encoding."""

    def __init__(self, categorical_columns: Optional[List[str]] = None, sparse: bool = False):
        if categorical_columns is None:
            from src.preprocessing.feature_config import CATEGORICAL_FEATURES
            categorical_columns = CATEGORICAL_FEATURES
        self.categorical_columns = categorical_columns
        self.encoder = OneHotEncoder(sparse_output=sparse, handle_unknown="ignore")
        self.encoded_feature_names_: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Fits the one-hot encoder on specified categorical columns."""
        existing_categorical = [col for col in self.categorical_columns if col in df.columns]
        if not existing_categorical:
            self.is_fitted = True
            return self

        self.encoder.fit(df[existing_categorical])
        self.encoded_feature_names_ = list(
            self.encoder.get_feature_names_out(existing_categorical)
        )
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical columns using the fitted one-hot encoder."""
        if not self.is_fitted:
            raise ValueError("The encoder has not been fitted yet. Call fit() before transform().")

        df_transformed = df.copy()
        existing_categorical = [col for col in self.categorical_columns if col in df.columns]

        if not existing_categorical:
            return df_transformed

        encoded_data = self.encoder.transform(df_transformed[existing_categorical])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.encoded_feature_names_,
            index=df_transformed.index,
        )

        # Drop original categorical columns and concatenate the encoded columns
        df_transformed = df_transformed.drop(columns=existing_categorical)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the DataFrame in one step."""
        return self.fit(df).transform(df)
