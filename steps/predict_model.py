from zenml import step
import pandas as pd
from sklearn.base import ClassifierMixin
from typing import Union

@step
def predict_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame
) -> pd.Series:
    """Use the trained model to make predictions on the test set."""
    predictions = model.predict(X_test)
    return pd.Series(predictions)