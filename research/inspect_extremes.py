import pandas as pd
import numpy as np

def inspect_extremes():
    df = pd.read_csv("train-test.csv")
    df['rpm'] = df['posted_rate'] / df['distance']
    
    print("=== ROWS WITH POSTED_RATE > 8000 ===")
    high_rates = df[df['posted_rate'] > 8000].sort_values('posted_rate', ascending=False)
    print(f"Total rows with posted_rate > 8000: {len(high_rates)}")
    print(high_rates[['load_id', 'date', 'pickup', 'delivery', 'distance', 'equipment', 'weight', 'quote_signal', 'market_index', 'posted_rate', 'rpm']].head(20))
    
    print("\n=== ROWS WITH RPM > 6.0 ===")
    high_rpm = df[df['rpm'] > 6.0].sort_values('rpm', ascending=False)
    print(f"Total rows with rpm > 6.0: {len(high_rpm)}")
    print(high_rpm[['load_id', 'date', 'pickup', 'delivery', 'distance', 'equipment', 'weight', 'quote_signal', 'market_index', 'posted_rate', 'rpm']].head(20))
    
    # Check if there are duplicate loads or corrupted quotes
    # Look at quote_signal vs rpm for these rows!
    print("\nCompare quote_signal and rpm for high rate rows:")
    print(high_rates[['posted_rate', 'distance', 'quote_signal', 'rpm']].head(10))

if __name__ == "__main__":
    inspect_extremes()
