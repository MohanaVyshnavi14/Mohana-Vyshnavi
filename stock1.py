# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Define the date range
start = '2010-01-01'
end = '2019-12-31'

# Fetch stock data for AAPL using yfinance
df = yf.download('AAPL', start=start, end=end)

# Reset index to ensure all columns are accessible
df = df.reset_index()

# Sorting the index to avoid the performance warning
df = df.sort_index(axis=1)

# Drop unnecessary columns safely
df = df.drop(['Date', 'Adj Close'], axis=1, errors='ignore')

# Display the first few rows to verify
print(df.head())

# Plot the 'Close' prices
plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label='AAPL Close Price', color='blue')
plt.title('AAPL Close Price Over Time (2010-2019)')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

# Calculate 100-period moving average
ma100 = df.Close.rolling(100).mean()

# Plot Close prices and 100-period moving average
plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.title('AAPL Close Price with 100-Period Moving Average')
plt.show()

# Calculate 200-period moving average
ma200 = df.Close.rolling(200).mean()

# Plot Close prices and moving averages
plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.title('AAPL Close Price with 100- and 200-Period Moving Averages')
plt.show()

# Splitting data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Check the shape of the data
print(data_training.shape)
print(data_testing.shape)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

# Prepare data for training
x_train = []
y_train = []

# Create training data based on 100 previous days of data
for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

# Convert lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Print the shape of x_train and y_train
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Define the model
model = Sequential()

# Add LSTM layers with Dropout
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50)
