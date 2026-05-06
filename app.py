import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor # <-- The new lightweight Neural Network
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
with col_ai:
    st.subheader("🤖 AI Forecast Model (MLP Neural Net)")
    if st.button("Generate Prediction"):
        with st.spinner("Training Neural Network..."):
            # Prepare Data
            data['Days'] = np.arange(len(data))
            X = data[['Days']].values
            y = data['Close'].values
            
            # Scale Data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            # Train Lightweight Neural Network
            # max_iter=500 ensures it learns enough patterns quickly
            mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
            mlp_model.fit(X_scaled, y_scaled)
            
            # Predict Next Day (Current total days + 1)
            next_day = np.array([[len(data)]])
            next_day_scaled = scaler_X.transform(next_day)
            
            prediction_scaled = mlp_model.predict(next_day_scaled)
            final_prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
            
            st.metric("Predicted Next Close", f"₹{final_prediction[0][0]:.2f}")
            st.caption("Model: Multi-Layer Perceptron (Artificial Neural Network)")
