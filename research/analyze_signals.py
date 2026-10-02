import pandas as pd
import numpy as np

def analyze_signals():
    df_train = pd.read_csv("train-test.csv")
    df_val = pd.read_csv("validation.csv")
    df_dec = pd.read_csv("december-chart-inputs.csv")
    
    print("=== CHECK LAT/LON CONSISTENCY FOR CITIES ===")
    for city in ['Lexington', 'Fort Wayne']:
        pts_train = df_train[df_train['pickup'] == city][['pickup_lat', 'pickup_lon']].drop_duplicates()
        pts_val = df_val[df_val['pickup'] == city][['pickup_lat', 'pickup_lon']].drop_duplicates()
        print(f"City {city} pickup lat/lon in train:\n{pts_train}")
        print(f"City {city} pickup lat/lon in val:\n{pts_val}")
        
    print("\n=== ARE MARKET_INDEX AND QUOTE_SIGNAL DATE-DEPENDENT OR LANE-DEPENDENT? ===")
    # Group by date: how many unique market_index per date?
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_val['date'] = pd.to_datetime(df_val['date'])
    
    daily_mi = df_train.groupby('date')['market_index'].agg(['count', 'nunique', 'min', 'max', 'std'])
    print("Daily market_index stats in train (first 10 days):\n", daily_mi.head(10))
    print("Overall daily market_index nunique distribution:\n", daily_mi['nunique'].value_counts())
    
    daily_qs = df_train.groupby('date')['quote_signal'].agg(['count', 'nunique', 'min', 'max', 'std'])
    print("Daily quote_signal stats in train (first 10 days):\n", daily_qs.head(10))
    print("Overall daily quote_signal nunique distribution:\n", daily_qs['nunique'].value_counts())
    
    # Check if market_index has a single value per date!
    daily_mi_all = df_train.dropna(subset=['market_index']).groupby('date')['market_index'].unique()
    print("Is market_index uniquely determined by date? Sample:\n", daily_mi_all.head(5))
    
    # Check if quote_signal is per lane, per equipment, per load?
    print("\nQuote signal correlation with rate per mile:")
    df_train['rpm'] = df_train['posted_rate'] / df_train['distance']
    print(df_train[['posted_rate', 'rpm', 'quote_signal', 'market_index', 'distance', 'weight']].corr())

    # Check December chart inputs: Does it have quote_signal or market_index?
    print("\nDecember chart input columns:", df_dec.columns.tolist())
    
    # Check if there are loads for Lexington -> Fort Wayne in train or val:
    lex_fw_train = df_train[(df_train['pickup'] == 'Lexington') & (df_train['delivery'] == 'Fort Wayne')]
    lex_fw_val = df_val[(df_val['pickup'] == 'Lexington') & (df_val['delivery'] == 'Fort Wayne')]
    print(f"Lexington -> Fort Wayne in train: {len(lex_fw_train)}, in val: {len(lex_fw_val)}")
    if len(lex_fw_train) > 0:
        print(lex_fw_train[['date', 'equipment', 'distance', 'weight', 'posted_rate', 'quote_signal', 'market_index']].head())

if __name__ == "__main__":
    analyze_signals()
