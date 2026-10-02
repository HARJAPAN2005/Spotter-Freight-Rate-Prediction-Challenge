import pandas as pd
import numpy as np

def analyze_market_index_and_date():
    df = pd.read_csv("train-test.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['rpm'] = df['posted_rate'] / df['distance']
    df['ratio'] = df['rpm'] / df['quote_signal']
    df['month'] = df['date'].dt.month
    
    print("Monthly market_index vs ratio:")
    monthly = df.groupby('month').agg({
        'market_index': ['mean', 'min', 'max'],
        'ratio': ['mean', 'median', 'std'],
        'quote_signal': ['mean'],
        'posted_rate': ['mean']
    })
    print(monthly)

    # Let's inspect daily market index and ratio
    daily = df.groupby('date').agg({
        'market_index': 'mean',
        'ratio': 'median'
    })
    print("\nCorrelation between daily mean market_index and daily median ratio:")
    print(daily.corr())
    
    # What about validation set? What does validation market_index look like?
    df_val = pd.read_csv("validation.csv")
    df_val['date'] = pd.to_datetime(df_val['date'])
    df_val['month'] = df_val['date'].dt.month
    print("\nValidation monthly market_index:")
    print(df_val.groupby('month')['market_index'].agg(['count', 'mean', 'min', 'max']))
    print(df_val.groupby('month')['quote_signal'].agg(['count', 'mean', 'min', 'max']))

if __name__ == "__main__":
    analyze_market_index_and_date()
