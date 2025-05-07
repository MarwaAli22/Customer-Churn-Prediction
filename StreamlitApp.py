import pandas as pd
import numpy as np
import joblib
import streamlit as st


# Title 
st.title("Customer Churn Prediction")
 
# Load dataset
st.write("This simple app uses a saved naive bayes model to predict customer churn")

df = pd.read_excel('churn_dataset.xlsx')
st.subheader("📉Dataset review")
st.write(df.head())

# Load model
model = joblib.load('naivebayesClassifier.pkl')

# Sidebar 
st.sidebar.header("Input customer features").write("Enter customer details to predict if will churn or not")
# Slider Input features for the user
age =st.sidebar.slider('age', 
                       int(df['Age'].min()), 
                       int(df['Age'].max()),
                       int(df['Age'].mean())
                       )
tenure = st.sidebar.slider('Tenure',
                            int(df['Tenure'].min()),
                            int(df['Tenure'].max()),
                            int(df['Tenure'].mean())
                            )
gender = st.sidebar.selectbox("Gender",
                       ("Male" ,"Female")
                       )

# Convert Gender to numerical value
if gender == "Male":
    gender = 1 
else :
    gender = 0

# Prediction
input_data = np.array([[age, tenure, gender]])
pred_satuts = {0:'NOT churn',
               1:'WILL churn'
               }

prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# Show prediction probabilities
st.subheader("📊prediction probability")
st.write(f"probability of NOT churn: {prediction_proba[0]:.2%}")
st.write(f"probability of WILL churn: {prediction_proba[1]:.2%}")
 
