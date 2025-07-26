from pipeline.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    # Define your CSV file path and target column
    data_path = "data/Iris.csv"
    target_col = "Species"

    # Run the ZenML pipeline
    run_training_pipeline(
        file_path=data_path,
        target_col=target_col
    )