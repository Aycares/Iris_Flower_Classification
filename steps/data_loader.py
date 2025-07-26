from zenml import step
import pandas as pd

@step
def load_data(file_path: str) -> pd.DataFrame:
    """
    ZenML step to load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")