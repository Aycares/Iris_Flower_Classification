from sklearn.ensemble import RandomForestClassifier
from pipeline import run_training_pipeline

if __name__ == "__main__":
    # Define your CSV file path and target column
    data_path = (r'.\Iris.csv')  # or "Iris.csv" if it's in the same folder
    target_col = "Species"

    # Run the ZenML pipeline
    run_training_pipeline(
        file_path=data_path,
        target_col=target_col
    )

