from zenml import step
from typing import Any
from sklearn.metrics import classification_report

@step
def evaluate_model(predictions: Any, y_test: Any) -> None:
    """Evaluate the model using classification metrics."""
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, predictions))