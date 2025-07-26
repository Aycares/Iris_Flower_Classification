from zenml import step
from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

from schema import TARGET_COLUMN, DROP_COLUMNS, FEATURE_COLUMNS


@step
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess the dataset: drop unused columns, split features and target, then split train/test."""
    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLUMNS)

    # Extract features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    return X_train, X_test, y_train, y_test