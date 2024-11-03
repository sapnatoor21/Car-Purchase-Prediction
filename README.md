# Car-Purchase-Prediction Web App

This repository contains a web application built using Streamlit that predicts whether an individual is likely to purchase a car based on their age and annual income. The app employs a logistic regression model trained on a sample dataset.

Features
---
User Input Form: Users can enter their age and annual income through a simple form.

Model Training: The app uses a logistic regression model trained on a dataset containing age, income, and purchase status.

Prediction Output: After submitting the form, users receive a prediction indicating whether they are likely to purchase a car.

Model Accuracy Display: The app also displays the accuracy of the model based on a test dataset.

Requirements
---
To run this application, you'll need:

Python 3.x

Streamlit

Pandas

Scikit-learn

You can install the required libraries using pip:

pip install streamlit, pandas, scikit-learn

How It Works
---
Data Preparation: A sample dataset is created with features (Age, Income) and a target variable (Purchased).

Model Training: The data is split into training and testing sets. A logistic regression model is trained using the training set.

Prediction: When the user inputs their details, the app uses the trained model to predict whether they are likely to purchase a car and displays the result.
