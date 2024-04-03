import pandas as pd
import talib
from get_data import get_data
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
def calced(file_path,filename):
    data = pd.read_csv(file_path)
    if not filename.endswith('.csv'):
        return False
    if 'Doji' in data.columns:
        # có doji => ko cho chạy
        print("=========shouldn't run==========")
        return False
    else:
        print("===========should run===========")
        return True
            
def calc(path = '../Data'):
    dfs = []
    # Iterate over files in the folder
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if calced(file_path=file_path,filename=filename):
        
            # Read CSV file into DataFrame and append to the list
            data = pd.read_csv(file_path)
            data.dropna(axis=0, how='all', inplace=True)
    
    
            # Sort data by date if it's not already sorted
            data = data.sort_values(by='timestamp_o')
            

            # # Trend indicator (OVERLAP STUDIES)
                # Calculate Moving Average (MA)
            data['MA'] = talib.EMA(data['cl'], timeperiod=5)
            data['MA'] = talib.EMA(data['cl'], timeperiod=10)
            data['MA'] = talib.EMA(data['cl'], timeperiod=20)
            data['MA'] = talib.EMA(data['cl'], timeperiod=34) 
            data['MA'] = talib.SMA(data['cl'], timeperiod=50)
            data['MA'] = talib.EMA(data['cl'], timeperiod=89) 
            data['MA'] = talib.SMA(data['cl'], timeperiod=100)
            data['MA'] = talib.EMA(data['cl'], timeperiod=200)

                # Simple Moving Average
            data['SMA'] = talib.SMA(data['cl'], timeperiod=10)
            data['SMA'] = talib.SMA(data['cl'], timeperiod=20) 
            data['SMA'] = talib.SMA(data['cl'], timeperiod=50) 
            data['SMA'] = talib.SMA(data['cl'], timeperiod=100)

                # Exponential Moving Average
            data['EMA'] = talib.EMA(data['cl'], timeperiod=10)
            data['EMA'] = talib.EMA(data['cl'], timeperiod=12)
            data['EMA'] = talib.EMA(data['cl'], timeperiod=20) 
            data['EMA'] = talib.SMA(data['cl'], timeperiod=50) 
            data['EMA'] = talib.SMA(data['cl'], timeperiod=100)

                # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(data['cl'])
            data['BB_Upper'] = upper
            data['BB_Middle'] = middle
            data['BB_Lower'] = lower

            # # MOMENTUM INDICATORS
                # Calculate Relative Strength Index (RSI)
            data['RSI'] = talib.RSI(data['cl'], timeperiod=14)

                # Calculate MACD (Moving Average Convergence Divergence)
            macd, signal, _ = talib.MACD(data['cl'])
            data['MACD'] = macd
            data['Signal'] = signal

                # Calculate ADX (Average Directional Index)
            data['ADX'] = talib.ADX(data['h'], data['l'], data['cl'], timeperiod=14)

            # # Volume Indicator:
            #     OBV
            data['OBV'] = talib.OBV(data['cl'], data['vol'])

            # # Volality Indicator:
            #     ATR                  
            data['ATR'] = talib.ATR(data['h'], data['l'], data['cl'], timeperiod=14)

            # # Pattern Recognition
                # Doji
            data['Doji'] = talib.CDLDOJI(data['o'], data['h'], data['l'], data['cl'])
                # Dragonfly Doji
            data['Dragonfly_Doji'] = talib.CDLDRAGONFLYDOJI(data['o'], data['h'], data['l'], data['cl'])
                # Gravestone Doji
            data['Gravestone_Doji'] = talib.CDLGRAVESTONEDOJI(data['o'], data['h'], data['l'], data['cl'])
                # Marubozu
            data['Marubozu'] = talib.CDLMARUBOZU(data['o'], data['h'], data['l'], data['cl'])
                # Harami Pattern
            data['Harami'] = talib.CDLHARAMI(data['o'], data['h'], data['l'], data['cl'])
                # Hammer
            data['Hammer'] = talib.CDLHAMMER(data['o'], data['h'], data['l'], data['cl'])
                # Inverted Hammer
            data['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(data['o'], data['h'], data['l'], data['cl'])
                # Engulfing
            data['Engulfing'] = talib.CDLENGULFING(data['o'], data['h'], data['l'], data['cl'])
                # Morning Star
            data['Morning_Star'] = talib.CDLMORNINGSTAR(data['o'], data['h'], data['l'], data['cl'])
                # Evening Star
            data['Evening_Star'] = talib.CDLEVENINGSTAR(data['o'], data['h'], data['l'], data['cl'])
                # Hanging Man
            data['Hanging_Man'] = talib.CDLHANGINGMAN(data['o'], data['h'], data['l'], data['cl'])
                # Dark Cloud Cover
            data['Dark_Cloud_Cover'] = talib.CDLDARKCLOUDCOVER(data['o'], data['h'], data['l'], data['cl'])



            patterns = ['Doji', 'Dragonfly_Doji', 'Gravestone_Doji', 'Marubozu', 'Harami', 'Hammer', 
                        'Inverted_Hammer', 'Engulfing', 'Morning_Star', 'Evening_Star', 'Hanging_Man', 'Dark_Cloud_Cover']

            dfs.append(data)
            data.to_csv(file_path, index=True)

    
    if len(dfs) != 0:    
        data = pd.concat(dfs, ignore_index=True)
        print(data)
        data.rename(columns={
            'o': 'Open',
            'cl': 'Close',
            'h': 'High',
            'l': 'Low',
            'vol': 'Volume'
            # Add more mappings for other columns as needed
        }, inplace=True)

        # data = data.head(150)
        # print(data.tail(500)['Morning_Star'])
        # Plot candlestick chart with identified patterns
        mpf.plot(data, type='candle', style='charles', volume=True, title='Candlestick Chart with Identified Patterns', 
                ylabel='Price', ylabel_lower='Volume', figscale=1.5, addplot=[
                    mpf.make_addplot(data['Hammer'], scatter=True, markersize=100, marker='|', color='green'),
                    mpf.make_addplot(data['Inverted_Hammer'], scatter=True, markersize=100, marker='|', color='red'),
                    ])
    
            

calc()

# # Plotting
# plt.figure(figsize=(10, 6))
# # plt.plot(data['cl'], label='cl Price', color='green')
# # plt.plot(upper, label='Upper BB', color='blue')
# # plt.plot(middle, label='Middle BB', color='cyan')
# # plt.plot(lower, label='Lower BB', color='blue')
# # plt.title('Bollinger Bands')

# plt.plot(data['MACD'], label='MACD', color='red')
# plt.plot(data['Signal'], label='Signal', color='blue')
# plt.title('MACD')

# # Adding title and legend
# plt.legend()

# # Show plot
# plt.show()