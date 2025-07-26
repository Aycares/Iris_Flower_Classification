from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import preprocess_data
from steps.data_training import train_model
from steps.predict_model import predict_model
from steps.evaluate_model import evaluate_model

@pipeline(name="iris_pipeline")
def training_pipeline(file_path: str, target_col: str):
    # Step 1: Load data
    df = load_data(file_path=file_path)

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df=df)

    # Step 3: Train model (only needs X_train and y_train)
    model = train_model(X_train=X_train, y_train=y_train)

    # Step 4: Predict (use model and X_test)
    predictions = predict_model(model=model, X_test=X_test)

    # Step 5: Evaluate predictions
    evaluate_model(predictions=predictions, y_test=y_test)

# Function to run the pipeline
def run_training_pipeline(file_path: str, target_col: str):
    training_pipeline(file_path=file_path, target_col=target_col)