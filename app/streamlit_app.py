import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
from src.predict import predict_from_input
from src.config import REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")

hours_studied = st.number_input("Hours Studied", min_value=0, max_value=100, value=10)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
assignments_completed = st.selectbox("Assignments Completed", ["Yes", "No"])

if st.button("Predict"):
    input_data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Previous_Scores": previous_scores,
        "Assignments_Completed": assignments_completed
    }

    try:
        result = predict_from_input(REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, input_data)

        st.success(f"Predicted Exam Score: {result['predicted_exam_score']}")

        # Colored Pass/Fail
        if result["predicted_pass_fail"] == "Pass":
            st.markdown("<h3 style='color:green;'>✅ PASS</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>❌ FAIL</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Train models first: python main.py")