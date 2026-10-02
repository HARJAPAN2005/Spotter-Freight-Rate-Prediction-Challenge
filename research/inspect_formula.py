import pandas as pd
import numpy as np

def inspect_formula():
    df = pd.read_csv("train-test.csv")
    
    # Let's filter to rows where ratio is almost exactly 1.0
    # In month 1:
    m1 = df[df['date'].str.startswith('2025-01')].copy()
    m1['rpm'] = m1['posted_rate'] / m1['distance']
    m1['diff'] = m1['rpm'] - m1['quote_signal']
    m1['ratio'] = m1['rpm'] / m1['quote_signal']
    
    print("Month 1 diff describe:")
    print(m1['diff'].describe())
    
    print("Month 1 ratio describe:")
    print(m1['ratio'].describe())
    
    # Let's look at the first 20 rows of Month 1:
    print(m1[['load_id', 'distance', 'equipment', 'weight', 'quote_signal', 'market_index', 'posted_rate', 'rpm', 'diff', 'ratio']].head(20))

if __name__ == "__main__":
    inspect_formula()
