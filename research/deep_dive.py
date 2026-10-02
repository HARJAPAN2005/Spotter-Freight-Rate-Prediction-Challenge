import pandas as pd
import numpy as np

def inspect_signals_in_depth():
    df_train = pd.read_csv("train-test.csv")
    df_val = pd.read_csv("validation.csv")
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_val['date'] = pd.to_datetime(df_val['date'])
    
    # 1. How is quote_signal determined?
    # Is quote_signal a function of lane, equipment, distance, etc.?
    # Or is quote_signal an algorithmic quote / broker rate estimate?
    print("=== QUOTE SIGNAL ANALYSIS ===")
    print("Quote signal correlation with features in train:")
    for col in ['distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
        print(f"  {col}: {df_train['quote_signal'].corr(df_train[col]):.4f}")
    
    print("\nQuote signal by equipment:")
    print(df_train.groupby('equipment')['quote_signal'].describe())
    
    # Let's check distance vs quote_signal
    # Often in freight, rate per mile decays with distance: rpm = a + b / sqrt(distance) or power law
    df_train['inv_dist'] = 1.0 / np.sqrt(df_train['distance'])
    print(f"Correlation between quote_signal and 1/sqrt(distance): {df_train['quote_signal'].corr(df_train['inv_dist']):.4f}")

    # Check for Lexington -> Fort Wayne:
    print("\nQuote signal for Lexington -> Fort Wayne in train and val:")
    lex_fw_all = pd.concat([
        df_train[(df_train['pickup'] == 'Lexington') & (df_train['delivery'] == 'Fort Wayne')],
        df_val[(df_val['pickup'] == 'Lexington') & (df_val['delivery'] == 'Fort Wayne')]
    ])
    print(lex_fw_all.groupby('equipment')['quote_signal'].describe())

    # 2. What about market_index?
    print("\n=== MARKET INDEX ANALYSIS ===")
    # Is market_index a daily macro index across all loads on that date?
    daily_val_mi = df_val.groupby('date')['market_index'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("Validation daily market_index (std dev within day):")
    print(daily_val_mi['std'].describe())
    
    daily_train_mi = df_train.groupby('date')['market_index'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nTrain daily market_index (std dev within day):")
    print(daily_train_mi['std'].describe())

    # 3. How does posted_rate relate to quote_signal and market_index?
    print("\n=== REGRESSION ON POSTED_RATE ===")
    # Let's see: posted_rate / distance vs quote_signal and market_index
    # or posted_rate vs distance * quote_signal * market_index?
    df_train_clean = df_train.dropna(subset=['market_index', 'weight']).copy()
    
    # Let's test a simple formula: base = distance * quote_signal
    df_train_clean['base'] = df_train_clean['distance'] * df_train_clean['quote_signal']
    df_train_clean['ratio_posted_base'] = df_train_clean['posted_rate'] / df_train_clean['base']
    
    print("posted_rate / (distance * quote_signal) describe:")
    print(df_train_clean['ratio_posted_base'].describe())
    
    print("Correlation of ratio_posted_base with market_index:", df_train_clean['ratio_posted_base'].corr(df_train_clean['market_index']))

if __name__ == "__main__":
    inspect_signals_in_depth()
