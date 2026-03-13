# 🎓 Student Performance Prediction (Regression + Classification)

This project predicts student performance using Machine Learning with a simple Streamlit UI.

## ✅ What this project does

Given student input:

- Hours_Studied  
- Attendance  
- Previous_Scores  
- Assignments_Completed (Yes/No)

It predicts:

1. **Exam_Score** (Regression)  
2. **Pass/Fail** (Classification)

---

## 📁 Project Structure

```text
student-performance-ml/
│
├── data/
│   └── raw/
│       └── student_performance.csv
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train_regression.py
│   ├── train_classification.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── regression_model.pkl
│   └── classification_model.pkl
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧾 Required CSV Columns

Your CSV must contain these columns exactly:

- `Hours_Studied`
- `Attendance`
- `Previous_Scores`
- `Assignments_Completed`  (Yes/No)
- `Exam_Score`
- `pass_fail`  (0/1)

### pass_fail rule
Create manually in Excel:

- `pass_fail = 1` if `Exam_Score >= 40`
- else `0`

Example formula:
```excel
=IF(Exam_Score_cell>=40,1,0)
```

---

## ⚙️ Installation (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is empty, use:

```txt
pandas
numpy
scikit-learn
streamlit
joblib
```

---

## 🚀 Run the project

### 1) Train models
```powershell
python main.py
```

This creates:

- `models/regression_model.pkl`
- `models/classification_model.pkl`

### 2) Run UI
```powershell
streamlit run app/streamlit_app.py
```

---

## 🧠 Models Used

- **Linear Regression** → Exam_Score prediction
- **Logistic Regression** → Pass/Fail prediction

---

## 🎨 UI Behavior

- Predicted score is displayed
- **PASS** is shown in **green**
- **FAIL** is shown in **red**

---

## ❗Common Errors

### `ModuleNotFoundError: No module named 'src'`
Already fixed in `streamlit_app.py` by adding project root to `sys.path`.

### Model file not found
Run:
```powershell
python main.py
```
before launching Streamlit.

### Column mismatch error
Check column names in CSV are exact (same spelling/case).

---

## 📌 Future Improvements

- Add model comparison (Decision Tree / Random Forest / XGBoost)
- Add feature importance chart
- Add downloadable prediction report
- Deploy app on Streamlit Cloud / Render
