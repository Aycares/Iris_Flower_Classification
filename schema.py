# schema.py

# The target column for classification
TARGET_COLUMN = "Species"

# All feature columns
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

# Columns to drop
DROP_COLUMNS = ["Id"]