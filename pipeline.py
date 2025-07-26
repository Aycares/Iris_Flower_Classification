from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import preprocess_data
from steps.model_training import train_model


@pipeline
def training_pipeline(file_path: str, target_col: str):
    # Step 1: Load the data
    df = load_data(file_path=file_path)

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df=df, target_col=target_col)

    # Step 3: Train the model
    train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run_training_pipeline(file_path: str, target_col: str):
    """Runs the ZenML training pipeline."""
    pipeline_instance = training_pipeline(
        file_path=file_path,
        target_col=target_col
    )
    pipeline_instance.run()
