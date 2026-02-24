# 🎓 Student Exam Score Predictor: Advanced ML Pipeline & Web App

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Advanced_ML-brightgreen?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

## 📖 Project Overview
The **Student Exam Score Predictor** is a complete, end-to-end Data Science project that leverages advanced gradient boosting frameworks to predict academic performance. By analyzing a student's demographics, academic habits, lifestyle, and environment, the model forecasts their final exam score.

This repository demonstrates professional machine learning workflows: dynamic data preprocessing, rigorous hyperparameter tuning (`GridSearchCV`) across competing advanced algorithms (LightGBM vs. XGBoost), pipeline serialization, and deployment via an interactive Streamlit web application.

## ✨ Key Features
*   **Dynamic Preprocessing:** Utilizes `make_column_selector` to automatically apply `StandardScaler` to numeric types and `OneHotEncoder` to categorical types, ensuring robust handling of new, unseen data.
*   **Algorithmic Benchmarking:** Pits two industry-standard gradient boosting regressors—**XGBoost** and **LightGBM**—against each other to find the optimal predictive engine.
*   **Hyperparameter Tuning:** Implements `GridSearchCV` to systematically search through learning rates, max depths, and estimator counts to minimize Mean Absolute Error (MAE).
*   **Production-Ready Pipelines:** Packages the winning model and preprocessing steps into a single, cohesive `.joblib` artifact to prevent data leakage and simplify deployment.
*   **Interactive Web UI:** A clean, responsive Streamlit dashboard categorized by *Personal Details*, *Lifestyle*, *Academic Habits*, and *Environment* for intuitive user interaction.

## 📊 Data Description
The model is trained on a dataset (`train.csv`) encompassing various factors that influence student success. 
*🔗 **Dataset Link:** [Insert Dataset Link Here]*

**Input Features:**
*   **Personal Details:** `age`, `gender`, `course` (e.g., B.Sc, BCA, B.Tech)
*   **Lifestyle:** `sleep_hours` (per day), `sleep_quality`
*   **Academic Habits:** `study_hours` (per day), `class_attendance` (%), `study_method` (e.g., Self-study, Online Videos)
*   **Environment:** `internet_access`, `facility_rating`, `exam_difficulty`

**Target Variable:**
*   `exam_score`: The final predicted numeric score of the student.

## 🛠️ Project Architecture

```text
├── train_model.ipynb                          # ML training script (Pipeline, GridSearchCV, Export)
├── app.py                                     # Streamlit web application interface
├── train.csv                                  # Raw training dataset [Not included, download required]
├── student_exam_predict_pipeline.joblib       # Serialized Scikit-Learn pipeline (Generated Output)
└── README.md                                  # Project documentation
```

## 🚀 Installation & Prerequisites

To run this project locally, ensure you have Python 3.8+ installed. 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SelimNajaf/Student-Exam-Predict/tree/main
   cd Student-Exam-Predict/tree/main/student_exam_predict
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install numpy pandas scikit-learn lightgbm xgboost joblib streamlit
   ```

3. **Add the Dataset:**
   Ensure the `train.csv` file is downloaded and placed in the root directory of the project before running the training script.

## 💻 Usage / How to Run

### Step 1: Train the Model & Export the Pipeline
Run the training script to preprocess the data, execute the Grid Search, benchmark the models, and generate the final `.joblib` model file.

```bash
python train_model.py
```
*Expected Terminal Output:*
> `🏆 Winning Model: LightGBM (MAE: 6.9853)`
> `Complete! The pipeline has been saved as 'student_exam_predict_pipeline.joblib'.`

### Step 2: Launch the Web Application
Once the pipeline is serialized, start the Streamlit web app to interact with the model.

```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`). Enter your hypothetical or real study habits to see the predicted exam score!

## 📈 Results / Model Evaluation

During the Cross-Validation Grid Search phase, both gradient boosting models performed exceptionally well, but **LightGBM** narrowly outperformed XGBoost on the test set:

*   **XGBoost MAE:** `6.9879` *(Optimal Params: LR 0.1, Max Depth 7, Estimators 300)*
*   **LightGBM MAE:** `6.9853` *(Optimal Params: LR 0.2, Max Depth 5, Estimators 300)*

Because LightGBM achieved the lowest Mean Absolute Error, the pipeline automatically selected it, retrained it on the *entire* dataset to maximize real-world accuracy, and serialized it for the Streamlit app.

## 🤝 Contributing
Contributions are highly encouraged! Whether it's adding SHAP value explanations to the Streamlit app, engineering new features, or optimizing the GridSearch:
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/AdvancedAnalytics`)
3. Commit your Changes (`git commit -m 'Add SHAP feature importance'`)
4. Push to the Branch (`git push origin feature/AdvancedAnalytics`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf-data-analyst](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*If you found this repository useful for learning advanced ML pipelines, please consider giving it a ⭐!*
