"""
Student Exam Score Predictor - Streamlit Application
A web interface allowing users to input their study habits and demographics
to predict their final exam score.
"""

import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    """Loads and caches the pre-trained machine learning pipeline."""
    return joblib.load('student_exam_predict_pipeline.joblib')

pipeline_model = load_model()

# ==========================================
# 2. USER INTERFACE HEADER
# ==========================================
st.title('🎓 Student Exam Score Predictor')
st.markdown('Fill in your academic and lifestyle details below to predict your likely exam score.')
st.divider()

# ==========================================
# 3. USER INPUT FORM (Organized in columns)
# ==========================================
# Using format_func to capitalize UI text while keeping raw data lowercase for the model
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")
    input_age = st.number_input('Age', min_value=17, max_value=45, value=22)
    input_gender = st.selectbox('Gender', ['female', 'male', 'other'], format_func=str.title)
    input_course = st.selectbox('Course', ['b.sc', 'diploma', 'bca', 'b.com', 'ba', 'bba', 'b.tech'], format_func=str.upper)
    
    st.subheader("💤 Lifestyle")
    input_sleep_hours = st.number_input('Sleep Hours (per day)', min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    input_sleep_quality = st.selectbox('Sleep Quality', ['average', 'poor', 'good'], format_func=str.title)

with col2:
    st.subheader("📚 Academic Habits")
    input_study_hours = st.number_input('Study Hours (per day)', min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    input_class_attendance = st.number_input('Class Attendance (%)', min_value=25, max_value=100, value=60)
    input_study_method = st.selectbox('Study Method', ['online videos', 'self-study', 'coaching', 'group study', 'mixed'], format_func=str.title)
    
    st.subheader("🏫 Environment")
    input_internet_access = st.selectbox('Internet Access', ['yes', 'no'], format_func=str.title)
    input_facility_rating = st.selectbox('Facility Rating', ['low', 'medium', 'high'], format_func=str.title)
    input_exam_difficulty = st.selectbox('Exam Difficulty', ['easy', 'moderate', 'hard'], format_func=str.title)

st.divider()

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
# Centered prediction button
if st.button('Predict Exam Score!', use_container_width=True):

    # Construct the input dataframe with exact feature names expected by the pipeline
    input_data = pd.DataFrame([{
        'age': input_age,
        'gender': input_gender,
        'course': input_course,
        'study_hours': input_study_hours,
        'class_attendance': input_class_attendance,
        'internet_access': input_internet_access,
        'sleep_hours': input_sleep_hours,
        'sleep_quality': input_sleep_quality,
        'study_method': input_study_method,
        'facility_rating': input_facility_rating,
        'exam_difficulty': input_exam_difficulty
    }])

    # Execute prediction
    prediction = pipeline_model.predict(input_data)
    
    # Display the result
    st.success(f'Based on your profile, your predicted exam score is: **{prediction[0]:.1f}**')