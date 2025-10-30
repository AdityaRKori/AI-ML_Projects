import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import holidays
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/content/GOOGL.csv')

# Incorporate holiday logic
us_holidays = holidays.US()
df['Date'] = pd.to_datetime(df['Date'])
df['Is_Holiday'] = df['Date'].apply(lambda date: date in us_holidays)

# Prepare data for candlestick chart
candlestick_df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
candlestick_df['Date'] = pd.to_datetime(candlestick_df['Date'])

# Select the 'Close' price for forecasting
data = df['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 60 # Using 60 days of data to predict the next day

# Create sequences
X, y = create_sequences(scaled_data, seq_length)

# Reshape X for LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=0)

# Make predictions
predictions = model.predict(X_test, verbose=0)

# Inverse transform predictions and actual values to original scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

# Prepare data for visualization
train_data = scaled_data[:train_size + seq_length]
valid_data = scaled_data[train_size:]

valid_df = df[train_size + seq_length:].copy()
valid_df['Predictions'] = predictions


# Streamlit App
st.header('Stock Price Forecasting Dashboard')

# Add a date range selector for historical data and predictions
st.subheader('Historical Stock Data and Predictions')
date_range_hist = st.date_input("Select date range for historical data and predictions:",
                                [candlestick_df['Date'].min().date(), candlestick_df['Date'].max().date()])

filtered_hist_df = candlestick_df.copy()
filtered_pred_df = valid_df.copy()

if len(date_range_hist) == 2:
    start_date_hist = date_range_hist[0]
    end_date_hist = date_range_hist[1]

    filtered_hist_df = candlestick_df[(candlestick_df['Date'] >= pd.to_datetime(start_date_hist)) &
                                      (candlestick_df['Date'] <= pd.to_datetime(end_date_hist))].copy()

    # Filter predictions based on the selected historical date range as well for combined visualization
    filtered_pred_df = valid_df[(valid_df['Date'] >= pd.to_datetime(start_date_hist)) &
                                (valid_df['Date'] <= pd.to_datetime(end_date_hist))].copy()


# Create the main figure
fig = go.Figure()

# Add the candlestick chart for historical data
fig.add_trace(go.Candlestick(x=filtered_hist_df['Date'],
                             open=filtered_hist_df['Open'],
                             high=filtered_hist_df['High'],
                             low=filtered_hist_df['Low'],
                             close=filtered_hist_df['Close'],
                             name='Historical'))

# Add the line chart for actual and predicted prices
fig.add_trace(go.Scatter(x=filtered_pred_df['Date'], y=filtered_pred_df['Close'], mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=filtered_pred_df['Date'], y=filtered_pred_df['Predictions'], mode='lines', name='Predicted Price'))


# Update layout for combined chart
fig.update_layout(title='Stock Price Historical Data and Predictions',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False) # Hide the range slider for clarity

st.plotly_chart(fig)

# Add a new section for model evaluation metrics
st.subheader("Model Evaluation Metrics")

# Display RMSE and MAE values
st.write(f"RMSE: {rmse}")
st.write(f"MAE: {mae}")
