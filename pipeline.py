from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import preprocess_data
from steps.data_training import train_model
import pandas as pd


@pipeline
def training_pipeline(file_path: str, target_col: str):
    # Step 1: Load the data
    df = load_data(file_path=file_path)

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df=df)

    # Step 3: Train the model
    train_model(X_train=X_train, y_train=y_train)


def run_training_pipeline(file_path: str, target_col: str):
    """Runs the ZenML training pipeline."""
    training_pipeline(
        file_path=file_path,
        target_col=target_col
    )