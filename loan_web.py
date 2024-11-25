import streamlit as st
from PIL import Image
import os
import pickle
import pandas as pd

# Paths for image and model
base_path = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(base_path, 'bank.png')
pickle_path = os.path.join(base_path, 'model.pkl')

@st.cache_data
def load_model():
    """
    Loads the trained machine learning model from the pickle file.
    
    Returns:
        model: The loaded machine learning model, or None if the file is not found.
    """
    try:
        return pickle.load(open(pickle_path, "rb"))
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'model.pkl' exists in the same directory as the script.")
        return None

# Load the model
model = load_model()

def run():
    """
    Runs the Streamlit application for bank loan prediction.
    Collects user inputs, preprocesses them, and predicts loan eligibility using the loaded model.
    """
    # Display logo
    try:
        img1 = Image.open(image_path)
        img1 = img1.resize((156, 145))  # Resize the image
        st.image(img1, use_container_width=False)  # Display image
    except FileNotFoundError:
        st.error("Image file not found. Ensure 'bank.png' exists in the same directory as the script.")

    st.subheader("Bank Loan Prediction using Machine Learning")  # App title

    # Input fields
    account_no = st.text_input("Account Number", max_chars=11)  # User's account number
    fn = st.text_input("Full Name", max_chars=250)  # User's full name

    # Gender selection
    gender_display = ("Female", "Male")
    gender = st.selectbox("Gender", range(len(gender_display)), format_func=lambda x: gender_display[x])

    # Marital status selection
    mar_display = ("No", "Yes")
    marital = st.selectbox("Married", range(len(mar_display)), format_func=lambda x: mar_display[x])

    # Dependents selection
    dep_display = ("No", "One", "Two", "More than Two")
    dependents = st.selectbox("Dependents", range(len(dep_display)), format_func=lambda x: dep_display[x])

    # Education level selection
    edu_display = ("Not Graduate", "Graduate")
    education = st.selectbox("Education", range(len(edu_display)), format_func=lambda x: edu_display[x])

    # Employment status selection
    emp_display = ("No", "Yes")
    employment = st.selectbox("Self Employed", range(len(emp_display)), format_func=lambda x: emp_display[x])

    # Applicant income input
    mon_income = st.number_input("Applicant's Monthly Income ($)", value=0, min_value=0)

    # Co-applicant income input
    co_mon_income = st.number_input("Co-Applicant's Monthly Income ($)", value=0, min_value=0)

    # Loan amount input
    loan_amount = st.number_input("Loan Amount", value=0, min_value=0)

    # Loan duration selection
    dur_display = ("2 Month", "6 Month", "8 Month", "1 Year", "16 Month")
    duration = st.selectbox("Loan Duration", range(len(dur_display)), format_func=lambda x: dur_display[x])
    duration_map = {0: 60, 1: 180, 2: 240, 3: 360, 4: 480}
    duration_value = duration_map.get(duration, 60)  # Map duration to numeric value

    # Property area selection
    area_display = ("Rural", "Semi-Urban", "Urban")
    area = st.selectbox("Property Area", range(len(area_display)), format_func=lambda x: area_display[x])

    # Predict loan eligibility on button click
    if st.button("Submit"):
        # Ensure mandatory fields are filled
        if not account_no or not fn:
            st.error("Please fill in all required fields (Account Number and Full Name).")
        elif model is None:
            st.error("The prediction model is not loaded. Please ensure the model file is available.")
        else:
            # Prepare input features
            values = [[
                gender, marital, dependents, education, employment,
                mon_income, co_mon_income, loan_amount, duration_value, area
            ]]
            
            # Align feature names with the trained model
            columns = ["Gender", "Married", "Dependents", "Education", "Self_Employed", 
                       "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
                       "Loan_Amount_Term", "Property_Area"]
            features = pd.DataFrame(values, columns=columns)

            # Predict loan eligibility
            try:
                prediction = model.predict(features)
                result = prediction[0]  # Extract the prediction result
                
                # Display the result
                if result == 0:
                    st.error(
                        f"Hello {fn}, Account Number: {account_no}. "
                        "Based on our calculations, you are not eligible for the loan."
                    )
                else:
                    st.success(
                        f"Hello {fn}, Account Number: {account_no}. "
                        "Congratulations! You are eligible for the loan."
                    )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Entry point
if __name__ == "__main__":
    run()
