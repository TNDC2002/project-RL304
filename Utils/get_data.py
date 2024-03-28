from binance.spot import Spot
import pandas as pd
from datetime import datetime
client = Spot()

last_timestamp = 1501549200000-1
i = 0
while i <= 35:
    print("start day: ",last_timestamp+1)
    df = pd.DataFrame(client.klines("BTCUSDT", "4h", startTime=last_timestamp+1), columns=['timestamp_o', 'o', 'h', 'l', 'cl', 'vol', 'timestamp_cl', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    last_timestamp = df['timestamp_cl'].iloc[-1]
    dt = datetime.fromtimestamp(last_timestamp / 1000.0)
    filename = dt.strftime('%Y-%m-%d_%H-%M-%S.csv')
    print("end day: ",filename)
    df.to_csv('../data/' + filename, index=False)
    i+=1
