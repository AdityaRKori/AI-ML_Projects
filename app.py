import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
# Assume the model file 'lung_cancer_model.pkl' is available
try:
    model = joblib.load('lung_cancer_model.pkl')
    model_loaded = True
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'lung_cancer_model.pkl' is in the same directory.")
    model_loaded = False

st.set_page_config(page_title="Lung Cancer Prediction App", layout="wide")

# Define pages
pages = {
    "Prediction": "prediction_page",
    "Dataset Information and WHO Advice": "info_page"
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Prediction Page
if selection == "Prediction":
    st.title("Lung Cancer Prediction")

    st.write("""
    This application predicts the likelihood of Lung Cancer based on the input features.
    Please provide the following information:
    """)

    # Create input widgets for the features
    gender = st.selectbox("GENDER", ['Male', 'Female'])
    age = st.slider("AGE", 20, 100, 50)
    smoking = st.selectbox("SMOKING", ['Yes', 'No'])
    yellow_fingers = st.selectbox("YELLOW_FINGERS", ['Yes', 'No'])
    anxiety = st.selectbox("ANXIETY", ['Yes', 'No'])
    peer_pressure = st.selectbox("PEER_PRESSURE", ['Yes', 'No'])
    chronic_disease = st.selectbox("CHRONIC DISEASE", ['Yes', 'No'])
    fatigue = st.selectbox("FATIGUE ", ['Yes', 'No'])
    allergy = st.selectbox("ALLERGY ", ['Yes', 'No'])
    wheezing = st.selectbox("WHEEZING", ['Yes', 'No'])
    alcohol_consuming = st.selectbox("ALCOHOL CONSUMING", ['Yes', 'No'])
    coughing = st.selectbox("COUGHING", ['Yes', 'No'])
    shortness_of_breath = st.selectbox("SHORTNESS OF BREATH", ['Yes', 'No'])
    swallowing_difficulty = st.selectbox("SWALLOWING DIFFICULTY", ['Yes', 'No'])
    chest_pain = st.selectbox("CHEST PAIN", ['Yes', 'No'])

    # Create a button to make predictions
    if st.button("Predict"):
        if model_loaded:
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'GENDER': [1 if gender == 'Male' else 0],
                'AGE': [age],
                'SMOKING': [1 if smoking == 'Yes' else 2],
                'YELLOW_FINGERS': [1 if yellow_fingers == 'Yes' else 2],
                'ANXIETY': [1 if anxiety == 'Yes' else 2],
                'PEER_PRESSURE': [1 if peer_pressure == 'Yes' else 2],
                'CHRONIC DISEASE': [1 if chronic_disease == 'Yes' else 2],
                'FATIGUE ': [1 if fatigue == 'Yes' else 2],
                'ALLERGY ': [1 if allergy == 'Yes' else 2],
                'WHEEZING': [1 if wheezing == 'Yes' else 2],
                'ALCOHOL CONSUMING': [1 if alcohol_consuming == 'Yes' else 2],
                'COUGHING': [1 if coughing == 'Yes' else 2],
                'SHORTNESS OF BREATH': [1 if shortness_of_breath == 'Yes' else 2],
                'SWALLOWING DIFFICULTY': [1 if swallowing_difficulty == 'Yes' else 2],
                'CHEST PAIN': [1 if chest_pain == 'Yes' else 2]
            })

            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Display the prediction result
            if prediction[0] == 1:
                st.error(f"Prediction: LUNG_CANCER: YES (Probability: {prediction_proba[0][1]:.2f})")
            else:
                st.success(f"Prediction: LUNG_CANCER: NO (Probability: {prediction_proba[0][0]:.2f})")
        else:
            st.warning("Model not loaded, cannot make predictions.")


# Dataset Information and WHO Advice Page
elif selection == "Dataset Information and WHO Advice":
    st.title("Dataset Information and WHO Advice")

    st.markdown("## Dataset Information")
    st.write("""
    The dataset used for this prediction model is the "Lung Cancer Dataset" from Kaggle,
    contributed by chandanmsr. It is a synthetic dataset containing 20000 entries
    with various features related to symptoms and habits, along with a target variable
    indicating the presence of Lung Cancer.

    **Features and their values:**
    - **GENDER**: 1 for Male, 0 for Female.
    - **AGE**: Age of the individual (integer).
    - **SMOKING**: 1 or 2 (representing different levels or aspects of smoking).
    - **YELLOW_FINGERS**: 1 or 2 (indicating presence or absence, or level).
    - **ANXIETY**: 1 or 2 (indicating presence or absence, or level).
    - **PEER_PRESSURE**: 1 or 2 (indicating presence or absence, or level).
    - **CHRONIC DISEASE**: 1 or 2 (indicating presence or absence, or level).
    - **FATIGUE**: 1 or 2 (indicating presence or absence, or level).
    - **ALLERGY**: 1 or 2 (indicating presence or absence, or level).
    - **WHEEZING**: 1 or 2 (indicating presence or absence, or level).
    - **ALCOHOL CONSUMING**: 1 or 2 (indicating presence or absence, or level).
    - **COUGHING**: 1 or 2 (indicating presence or absence, or level).
    - **SHORTNESS OF BREATH**: 1 or 2 (indicating presence or absence, or level).
    - **SWALLOWING DIFFICULTY**: 1 or 2 (indicating presence or absence, or level).
    - **CHEST PAIN**: 1 or 2 (indicating presence or absence, or level).
    - **LUNG_CANCER**: 1 for YES, 0 for NO (Target variable).

    *Note: The exact meaning of values 1 and 2 for some features might require
    referring to the dataset's original documentation if available. In this
    application, we use the numerical representation as used in the model training.*
    """)

    st.markdown("## WHO Advice")
    st.write("""
    Here is some general advice from the World Health Organization (WHO) regarding lung cancer:

    *   **Quit smoking:** Smoking is the leading cause of lung cancer. Quitting smoking is the most important step you can take to reduce your risk.
    *   **Avoid exposure to secondhand smoke:** Breathing in the smoke from others' cigarettes also increases your risk of lung cancer.
    *   **Avoid exposure to radon:** Radon is a radioactive gas that can accumulate in homes and cause lung cancer. Test your home for radon and take action to reduce levels if they are high.
    *   **Avoid exposure to other carcinogens:** In the workplace, avoid exposure to substances like asbestos, arsenic, chromium, and nickel, which can increase lung cancer risk.
    *   **Eat a healthy diet:** While not a direct preventative measure against lung cancer, a healthy diet contributes to overall health.
    *   **Be aware of the symptoms:** If you experience persistent coughing, chest pain, shortness of breath, wheezing, or unexplained weight loss, consult a doctor. Early detection can improve treatment outcomes.

    For more detailed information and advice, please refer to the official World Health Organization website.

    *Disclaimer: This information is for general knowledge and does not constitute medical advice. Always consult with a healthcare professional for any health concerns.*
    """)
