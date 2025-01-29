import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Set Streamlit title
st.title('Stock Trade Prediction with Candlestick Chart')

# Take user input for stock ticker symbol
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch stock data using yfinance
df = yf.download(user_input, start='2010-01-01', end='2019-12-31')

if df.empty:
    st.error(f"No data found for the ticker '{user_input}'. Please enter a valid stock ticker.")
    st.stop()

# Display raw data
st.subheader("Raw Data")
st.write(df.head())

# Ensure the index is datetime type and reset it for plotting
df.index = pd.to_datetime(df.index)
df.reset_index(inplace=True)  # Reset index to access Date as a column

# Check if essential columns are present and have valid data
required_columns = ['Open', 'High', 'Low', 'Close']
if not all(col in df.columns for col in required_columns):
    st.error("Essential columns (Open, High, Low, Close) are missing in the data.")
    st.stop()

if df[required_columns].isnull().any().any():
    st.error("Some essential columns contain null values. Cleaning data...")
    df = df.dropna(subset=required_columns)  # Drop rows with NaN values

# Display cleaned data
st.subheader("Cleaned Data")
st.write(df.head())

# Creating a candlestick figure
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
)])

# Update layout for better chart visualization
fig_candlestick.update_layout(
    title=f"{user_input} Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
    template="plotly_dark"
)

# Display the candlestick chart using Streamlit
st.subheader("Candlestick Chart")
st.plotly_chart(fig_candlestick)

# Debugging aids
st.subheader("Debugging Information")
st.write("Data Information:")
st.write(df.info())  # Display data info for debugging

st.write("Sample Candlestick Data Preview:")
st.write({
    'Dates': df['Date'].tolist()[:5],
    'Open Prices': df['Open'].tolist()[:5],
    'High Prices': df['High'].tolist()[:5],
    'Low Prices': df['Low'].tolist()[:5],
    'Close Prices': df['Close'].tolist()[:5],
})

# Test with sample data for debugging
st.subheader("Sample Candlestick Chart for Debugging")
sample_data = {
    'Date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
    'Open': [100, 102, 101, 103, 102],
    'High': [105, 107, 106, 108, 107],
    'Low': [95, 96, 97, 98, 96],
    'Close': [102, 104, 103, 105, 104],
}
df_sample = pd.DataFrame(sample_data)

fig_sample = go.Figure(data=[go.Candlestick(
    x=df_sample['Date'],
    open=df_sample['Open'],
    high=df_sample['High'],
    low=df_sample['Low'],
    close=df_sample['Close']
)])
st.plotly_chart(fig_sample)
