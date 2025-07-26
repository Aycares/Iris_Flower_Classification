# utils.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

def evaluate_classification_model(y_true, y_pred) -> dict:
    """Evaluate model predictions and return common classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1_score": f1_score(y_true, y_pred, average="macro"),
    }
    
    # Log the metrics
    logging.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")

    return metrics