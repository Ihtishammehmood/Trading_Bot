import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

class MT5DataIngest:
    def __init__(self):
        """Initialize MetaTrader5 connection"""
        if not mt5.initialize():
            print("MT5 Initialization failed!")
            mt5.shutdown()
            raise ConnectionError("Could not connect to MetaTrader 5")
        print('MT5 Initialized successfully')

    def get_data(self, symbol: str, timeframe: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data from MetaTrader5
        
        Args:
            symbol (str): Trading symbol (e.g., "EURUSD")
            timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_H4)
            start_date (str): Start date in "YYYY-MM-DD" format
            end_date (str): End date in "YYYY-MM-DD" format
        
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
        """
        df.to_csv(filename)
        print(f'Data saved to {filename} successfully')

    def __del__(self):
        """Cleanup MT5 connection on object destruction"""
        mt5.shutdown()


if __name__ == "__main__":
    mt5_data = MT5DataIngest()
    start = "2023-01-01"
    end = datetime.today().strftime("%Y-%m-%d")
    
    df = mt5_data.get_data("EURUSD", mt5.TIMEFRAME_H4, start, end)
    print(df.head())
    mt5_data.save_data(df, "EURUSD_H4.csv")