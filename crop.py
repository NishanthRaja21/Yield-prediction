import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("Crop Yield Prediction using Random Forest")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display first few rows
    st.write("### Dataset Preview")
    st.write(df.head())

    # Check if 'Yield' column exists
    if 'Yield' in df.columns:
        # Feature Engineering
        X = df.drop(columns=['Yield'])
        y = df['Yield']
        X = pd.get_dummies(X, columns=['Crop', 'Season', 'State'], drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display results
        st.write("### Model Performance Metrics")
        st.write(f"**Mean Absolute Error:** {mae:.4f}")
        st.write(f"**Mean Squared Error:** {mse:.4f}")
        st.write(f"**R2 Score:** {r2:.4f}")

        # Plot Actual vs Predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Yield")
        ax.set_ylabel("Predicted Yield")
        ax.set_title("Actual vs Predicted Yield")
        st.pyplot(fig)

    else:
        st.error("The dataset must contain a 'Yield' column.")

else:
    st.info("Please upload a CSV file to start the analysis.")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

