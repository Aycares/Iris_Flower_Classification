from zenml import step
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

@step
def evaluate_model(predictions: pd.Series, y_test: pd.Series) -> None:
    """Evaluate the model and log classification metrics."""
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"\nModel Evaluation Metrics:")
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")
    print(f"   • F1 Score:  {f1:.4f}")