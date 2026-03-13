import joblib
import pandas as pd


def predict_from_input(reg_model_path: str, clf_model_path: str, input_dict: dict):
    reg_model = joblib.load(reg_model_path)
    clf_model = joblib.load(clf_model_path)

    X_input = pd.DataFrame([input_dict])

    reg_pred = float(reg_model.predict(X_input)[0])
    clf_pred = int(clf_model.predict(X_input)[0])

    return {
        "predicted_exam_score": round(reg_pred, 2),
        "predicted_pass_fail": "Pass" if clf_pred == 1 else "Fail"
    }