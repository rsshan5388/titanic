import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('titanc_model.pkl')

# Sample user input (e.g., coming from a Streamlit form)
user_input = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # 1 = male
    'Age': [25],
    'Fare': [7.25],
    'Parch': [0],
    'SibSp': [0],
    'Embarked_C': [0],
    'Embarked_S': [1],
    'Embarked_Q': [0]
})

# Ensure the user input has the same columns as during training
# Get the columns used during training
train_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Ensure the input data has all the necessary columns, fill missing ones with 0s if needed
st.title('Titanic Survival Prediction')

# Display a brief description of the app
st.write("This app predicts whether a passenger survived the Titanic disaster based on the features provided.")

# User inputs for prediction
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
pclass = st.selectbox('Passenger Class', [1, 2, 3])
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
fare = st.number_input('Fare', min_value=0.0, value=7.25)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, value=0)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, value=0)

# Map input features to the correct format
sex = 1 if sex == 'Male' else 0

# Prepare the data for prediction (same as preprocessing)
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_C': [1 if embarked == 'C' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0],
})

# Scale the input data (use the same scaler as used for training)
input_scaled = scaler.transform(input_data)

# Predict the survival using the model
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    
    # Display prediction
    if prediction == 1:
        st.success('The passenger survived.')
    else:
        st.error('The passenger did not survive.')