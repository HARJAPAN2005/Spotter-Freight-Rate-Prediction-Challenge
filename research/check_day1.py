import pandas as pd
import numpy as np

def check_day1():
    df = pd.read_csv("train-test.csv")
    d1 = df[df['date'] == '2025-01-01'].copy()
    d1['rpm'] = d1['posted_rate'] / d1['distance']
    d1['diff'] = d1['rpm'] - d1['quote_signal']
    d1['ratio'] = d1['rpm'] / d1['quote_signal']
    
    print("Correlations on 2025-01-01 with diff:")
    for col in ['market_index', 'distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
        print(f"  {col}: {d1['diff'].corr(d1[col]):.4f}")
        
    print("\nCorrelations on 2025-01-01 with ratio:")
    for col in ['market_index', 'distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
        print(f"  {col}: {d1['ratio'].corr(d1[col]):.4f}")

    # Let's inspect 5 loads on 2025-01-01
    print("\nSample loads on 2025-01-01:")
    print(d1[['load_id', 'equipment', 'distance', 'weight', 'market_index', 'quote_signal', 'rpm', 'diff', 'ratio']].head(10))

if __name__ == "__main__":
    check_day1()
