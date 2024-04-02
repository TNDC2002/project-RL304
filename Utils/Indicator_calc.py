import pandas as pd
import talib
from Utils.get_data import get_data

data = get_data('../data')


# Sort data by date if it's not already sorted
data = data.sort_values(by='Timestamp')

# Calculate Moving Average (MA)
data['SMA'] = talib.SMA(data['Close'], timeperiod=20)  # Simple Moving Average
data['EMA'] = talib.EMA(data['Close'], timeperiod=20)  # Exponential Moving Average

# Calculate Relative Strength Index (RSI)
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

# Calculate MACD (Moving Average Convergence Divergence)
macd, signal, _ = talib.MACD(data['Close'])
data['MACD'] = macd
data['Signal'] = signal

# Calculate Bollinger Bands
upper, middle, lower = talib.BBANDS(data['Close'])
data['BB_Upper'] = upper
data['BB_Middle'] = middle
data['BB_Lower'] = lower

# Print the DataFrame with calculated indicators
print(data)
