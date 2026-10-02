import pandas as pd
import numpy as np

def test_relationship():
    df = pd.read_csv("train-test.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['rpm'] = df['posted_rate'] / df['distance']
    
    print("Correlation between rpm and quote_signal:")
    print(df[['rpm', 'quote_signal']].corr())
    
    df['diff_rpm_qs'] = df['rpm'] - df['quote_signal']
    df['ratio_rpm_qs'] = df['rpm'] / df['quote_signal']
    
    print("\nSummary of rpm / quote_signal:")
    print(df['ratio_rpm_qs'].describe())
    
    print("\nSummary of rpm - quote_signal:")
    print(df['diff_rpm_qs'].describe())

    print("\nWhat correlates with ratio_rpm_qs?")
    numeric_cols = ['market_index', 'weight', 'distance', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']
    for col in numeric_cols:
        print(f"Correlation with {col}: {df['ratio_rpm_qs'].corr(df[col]):.4f}")
        
    print("\nGroup ratio_rpm_qs by equipment:")
    print(df.groupby('equipment')['ratio_rpm_qs'].describe())

    print("\nGroup ratio_rpm_qs by month:")
    df['month'] = df['date'].dt.month
    print(df.groupby('month')['ratio_rpm_qs'].describe())
    
    print("\nLet's check if posted_rate = distance * quote_signal * f(...) or posted_rate = distance * quote_signal + ...:")
    # Let's inspect rows where diff or ratio is small or large
    print("\nTop 5 rows:")
    print(df[['distance', 'posted_rate', 'quote_signal', 'market_index', 'weight', 'equipment', 'rpm', 'ratio_rpm_qs', 'diff_rpm_qs']].head(10))

if __name__ == "__main__":
    test_relationship()
