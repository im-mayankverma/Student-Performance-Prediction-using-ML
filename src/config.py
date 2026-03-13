DATA_PATH = "data/raw/student_performance.csv"

FEATURE_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Assignments_Completed"
]

REGRESSION_TARGET = "Exam_Score"
CLASSIFICATION_TARGET = "pass_fail"

TEST_SIZE = 0.2
RANDOM_STATE = 42

REGRESSION_MODEL_PATH = "models/regression_model.pkl"
CLASSIFICATION_MODEL_PATH = "models/classification_model.pkl"