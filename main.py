from src.config import (
    DATA_PATH, FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET,
    TEST_SIZE, RANDOM_STATE, REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH
)
from src.preprocess import preprocess_and_split
from src.train_regression import train_regression_model
from src.train_classification import train_classification_model


def main():
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = preprocess_and_split(
        file_path=DATA_PATH,
        feature_cols=FEATURE_COLUMNS,
        regression_target=REGRESSION_TARGET,
        classification_target=CLASSIFICATION_TARGET,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    _, reg_metrics = train_regression_model(
        X_train, y_reg_train, X_test, y_reg_test, REGRESSION_MODEL_PATH
    )

    _, clf_metrics = train_classification_model(
        X_train, y_clf_train, X_test, y_clf_test, CLASSIFICATION_MODEL_PATH
    )

    print("\n=== Regression Metrics ===")
    for k, v in reg_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Classification Metrics ===")
    for k, v in clf_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()