import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    ZenML step to train a Random Forest Classifier.
    """
    try:
        model = RandomForestClassifier(random_state=23,max_depth=5,n_estimators=100)
        model.fit(X_train, y_train)
        logger.info("Random Forest training completed.")
        return model
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise RuntimeError(f"Model training failed: {e}")
    
    return model