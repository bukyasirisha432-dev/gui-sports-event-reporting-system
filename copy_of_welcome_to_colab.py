import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

start = '2012-01-01'
end = '2022-12-21'
stock = 'GOOG'

data = yf.download(stock, start=start, end=end)
data.reset_index(inplace=True)

ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

plt.figure(figsize=(10, 6))
plt.plot(data['Close'], 'g', label='Closing Price')
plt.plot(ma_100_days, 'r', label='100-Day MA')
plt.plot(ma_200_days, 'b', label='200-Day MA')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{stock} Price with Moving Averages')
plt.show()
data.dropna(inplace=True)
train_size = int(len(data) * 0.8)
data_train = data['Close'][:train_size]
data_test = data['Close'][train_size:]
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train.values.reshape(-1, 1))
x_train, y_train = [], []
for i in range(100, len(data_train_scaled)):
    x_train.append(data_train_scaled[i-100:i])
    y_train.append(data_train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
past_100_days = data_train[-100:].values.reshape(-1, 1)
data_test_scaled = scaler.transform(np.concatenate((past_100_days, data_test.values.reshape(-1, 1))))

x_test, y_test = [], []
for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

y_pred = model.predict(x_test)
scaling_factor = 1 / scaler.scale_
y_pred = y_pred * scaling_factor
y_test = y_test * scaling_factor
plt.figure(figsize=(12, 8))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{stock} Original vs Predicted Prices')
plt.legend()
plt.show()
