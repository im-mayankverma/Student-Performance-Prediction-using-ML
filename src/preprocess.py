import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def preprocess_and_split(
    file_path: str,
    feature_cols: list,
    regression_target: str,
    classification_target: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    df = load_data(file_path)

    # clean column names
    df.columns = [c.strip() for c in df.columns]

    # validate columns
    required = feature_cols + [regression_target, classification_target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # keep only needed columns
    df = df[required].copy()

    # normalize Assignments_Completed to Yes/No (if typed weird)
    df["Assignments_Completed"] = (
        df["Assignments_Completed"]
        .astype(str)
        .str.strip()
        .str.capitalize()
        .replace({"Y": "Yes", "N": "No", "1": "Yes", "0": "No", "True": "Yes", "False": "No"})
    )

    # convert targets
    df[regression_target] = pd.to_numeric(df[regression_target], errors="coerce")
    df[classification_target] = pd.to_numeric(df[classification_target], errors="coerce")

    # drop rows with missing target
    df = df.dropna(subset=[regression_target, classification_target])

    # features / targets
    X = df[feature_cols]
    y_reg = df[regression_target]
    y_clf = df[classification_target].astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test