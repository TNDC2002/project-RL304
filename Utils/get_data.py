from binance.spot import Spot
import os
import pandas as pd
from datetime import datetime

def RequestAndSavedata():
    client = Spot()
    last_timestamp = 1501549200000-1
    i = 0
    while i <= 35:
        print("start day: ",last_timestamp+1)
        df = pd.DataFrame(client.klines("BTCUSDT", "4h", startTime=last_timestamp+1), columns=['timestamp_o', 'o', 'h', 'l', 'cl', 'vol', 'timestamp_cl', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        # Convert 'timestamp_o' column to datetime format (assuming it's in milliseconds)
        
        
        
        last_timestamp = df['timestamp_cl'].iloc[-1]
        dt = datetime.fromtimestamp(last_timestamp / 1000.0)
        filename = dt.strftime('%Y-%m-%d_%H-%M-%S.csv')
        print("end day: ",filename)
        df.to_csv('../Data/' + filename, index=False)
        i+=1

def get_data(folder_path = '../Data/'):
    # Initialize an empty list to store DataFrames
    dfs = []
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct full path to the file
            file_path = os.path.join(folder_path, filename)
            
            # Read CSV file into DataFrame and append to the list
            df = pd.read_csv(file_path)
            df.dropna(axis=0, how='all', inplace=True)
            dfs.append(df)

    # Merge DataFrames into a single DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

# Example usage:
# folder_path = '../Data'
# merged_df = get_data(folder_path)
# print(merged_df)
