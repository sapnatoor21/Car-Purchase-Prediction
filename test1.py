# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the web app
st.title("Car Purchase Prediction")

# Create a form to input data
with st.form("input_form"):
    age = st.number_input("Enter your age:", min_value=18, max_value=100)
    income = st.number_input("Enter your annual income ($):", min_value=10000, max_value=1000000)
    submit_button = st.form_submit_button("Predict")

# Create a sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28],
    'Income': [50000, 55000, 65000, 70000, 75000, 80000, 60000, 70000, 75000, 55000, 40000, 45000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

# Prepare features and target variable
X = df[['Age', 'Income']]
y = df['Purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set and display accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Make a prediction based on user input
if submit_button:
    user_data = pd.DataFrame({'Age': [age], 'Income': [income]})
    prediction = model.predict(user_data)[0]
    prediction_text = "likely to purchase a car." if prediction == 1 else "not likely to purchase a car."
    st.write(f"The model predicts that this user is {prediction_text}")
