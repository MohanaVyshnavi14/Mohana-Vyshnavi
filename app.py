import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime
import numpy as np

# Download NLTK Data
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Apply Custom Styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .block-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #0b5a3a;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0a4731;
    }
    h1, h2, h3, h4 {
        color: #2e3b4e;
    }
    .stDataFrame {
        border: 2px solid #dcdcdc;
        border-radius: 5px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #f2f2f2;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0b5a3a;
    }
    .stSelectbox select {
        background-color: #f2f2f2;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("\U0001F4C8 Stock Analysis and Sentiment Analysis App")

# Sidebar Inputs
st.sidebar.header("User Input")
tickers = st.sidebar.text_input("Enter Stock Ticker(s) (comma separated)", value="AAPL,GOOGL").strip().upper()
tickers = tickers.split(",")

start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

if start_date > end_date:
    st.sidebar.error("Start date must be earlier than end date.")

# Fetch Stock Data
st.subheader("\U0001F4C9 Stock Data")
try:
    if tickers:
        dfs = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                st.warning(f"No data retrieved for {ticker}. Please check the ticker symbol and date range.")
            else:
                dfs[ticker] = df
                st.write(f"Showing data for {ticker}:")
                st.dataframe(df.tail())
    else:
        st.warning("Please enter a stock ticker to fetch data.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

# Dividend Yield (Dividend Ratio) for Tickers
st.subheader("\U0001F4B0 Dividend Yield & Ratio")
for ticker in tickers:
    try:
        stock_info = yf.Ticker(ticker).info
        dividend_yield = stock_info.get('dividendYield', 'No dividend data available')
        dividend_ratio = stock_info.get('dividendRate', 'No dividend ratio data available')

        if dividend_yield != 'No dividend data available':
            st.write(f"{ticker}: Dividend Yield: {dividend_yield * 100:.2f}%")
        else:
            st.write(f"{ticker}: Dividend Yield: {dividend_yield}")

        if dividend_ratio != 'No dividend ratio data available':
            st.write(f"{ticker}: Dividend Ratio: {dividend_ratio}")
        else:
            st.write(f"{ticker}: Dividend Ratio: {dividend_ratio}")

    except Exception as e:
        st.warning(f"Error retrieving dividend data for {ticker}: {e}")

# Technical Analysis
if 'dfs' in locals() and dfs:
    st.subheader("\U0001F4C8 Technical Analysis")

    for ticker, df in dfs.items():
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RollingStd'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['RollingStd'] * 2)
        df['Lower'] = df['MA20'] - (df['RollingStd'] * 2)

        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        df['RS'] = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + df['RS']))

        # Plot Bollinger Bands
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'], label=f'{ticker} Close Price', color='blue')
        ax.plot(df['MA20'], label='20-Day MA', color='orange')
        ax.plot(df['Upper'], label='Upper Bollinger Band', color='green', linestyle='--')
        ax.plot(df['Lower'], label='Lower Bollinger Band', color='red', linestyle='--')
        ax.set_title(f"Bollinger Bands for {ticker}")
        ax.legend()
        st.pyplot(fig)

        # Plot RSI
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['RSI'], label=f'{ticker} RSI', color='purple')
        ax.axhline(70, color='red', linestyle='--', label='Overbought Threshold')
        ax.axhline(30, color='green', linestyle='--', label='Oversold Threshold')
        ax.set_title(f"RSI for {ticker}")
        ax.legend()
        st.pyplot(fig)

# Risk Analysis (VaR and CVaR)
if 'dfs' in locals() and dfs:
    st.subheader("\U0001F4C9 Risk Analysis (VaR and CVaR)")

    confidence_level = st.sidebar.slider("Select Confidence Level for VaR/CVaR", min_value=90, max_value=99, value=95, step=1)

    for ticker, df in dfs.items():
        if not df.empty:
            # Daily Returns
            df['Daily Return'] = df['Close'].pct_change()
            df.dropna(inplace=True)

            # VaR Calculation
            var = np.percentile(df['Daily Return'], 100 - confidence_level)
            cvar = df['Daily Return'][df['Daily Return'] <= var].mean()

            st.write(f"**{ticker} Risk Metrics:**")
            st.write(f"- Value at Risk (VaR) at {confidence_level}% confidence: {var:.2%}")
            st.write(f"- Conditional Value at Risk (CVaR): {cvar:.2%}")

            if var < -0.03:
                st.warning(f"High Risk Alert: VaR for {ticker} exceeds acceptable threshold.")

            # Plot Daily Returns Distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(df['Daily Return'], kde=True, bins=50, ax=ax, color='blue')
            ax.axvline(var, color='red', linestyle='--', label=f'VaR ({confidence_level}%)')
            ax.axvline(cvar, color='orange', linestyle='--', label='CVaR')
            ax.set_title(f"Daily Returns Distribution for {ticker}")
            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

# Sentiment Analysis
st.subheader("\U0001F4DC Sentiment Analysis")
input_text = st.text_area("Enter text data (e.g., news headlines or tweets) for sentiment analysis:", "")
if input_text:
    # Check if input text is understandable
    if input_text.strip() == "":
        st.error("Please enter some valid text for sentiment analysis.")
    else:
        # Process Input Text
        sentiments = [sentiment_analyzer.polarity_scores(line)['compound'] for line in input_text.split('\n')]

        # Add sentiment labels based on compound score
        sentiment_labels = []
        for score in sentiments:
            if score >= 0.05:
                sentiment_labels.append("Positive")
            elif score <= -0.05:
                sentiment_labels.append("Negative")
            else:
                sentiment_labels.append("Neutral")

        sentiment_df = pd.DataFrame({
            'Text': input_text.split('\n'),
            'Sentiment Score': sentiments,
            'Sentiment Label': sentiment_labels  # Add sentiment labels to dataframe
        })

        # Plot Sentiment Scores
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Sentiment Score", y="Text", data=sentiment_df, ax=ax)
        ax.set_title("Sentiment Scores of Text Entries")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Text Entries")
        st.pyplot(fig)
else:
    st.info("Please enter some text above to analyze sentiment.")

# Call Option Data
st.subheader("\U0001F4B8 Call Option Data")
call_option_ticker = st.text_input("Enter Stock Ticker for Call Option Analysis (e.g., AAPL)", value="AAPL").strip().upper()

if call_option_ticker:
    try:
        stock_data = yf.Ticker(call_option_ticker)
        options_data = stock_data.options
        if options_data:
            st.write(f"Available Expiry Dates for {call_option_ticker}:")
            st.write(options_data)

            # Fetch and display options data for a selected expiry date
            selected_expiry = st.selectbox("Select Expiry Date", options_data)
            option_chain = stock_data.option_chain(selected_expiry)

            st.write(f"Call Options for {call_option_ticker} Expiring on {selected_expiry}:")
            st.dataframe(option_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'openInterest']])

        else:
            st.warning(f"No options data available for {call_option_ticker}.")
    except Exception as e:
        st.error(f"Error retrieving options data for {call_option_ticker}: {e}")
