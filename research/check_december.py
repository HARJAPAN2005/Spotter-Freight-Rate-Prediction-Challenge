import pandas as pd
import numpy as np

def check_december():
    df_val = pd.read_csv("validation.csv")
    df_dec = pd.read_csv("december-chart-inputs.csv")
    
    print("December chart inputs:")
    print(df_dec)
    
    df_val['date'] = pd.to_datetime(df_val['date'])
    val_dec = df_val[(df_val['date'] >= '2025-12-01') & (df_val['date'] <= '2025-12-31')]
    print(f"\nTotal loads in validation in December: {len(val_dec)}")
    print("Unique dates in validation December:", val_dec['date'].nunique())
    
    # Check if there are Lexington -> Fort Wayne loads in validation in December:
    lex_fw = val_dec[(val_dec['pickup'] == 'Lexington') & (val_dec['delivery'] == 'Fort Wayne')]
    print(f"Lexington -> Fort Wayne in val Dec: {len(lex_fw)}")
    print(lex_fw[['load_id', 'date', 'equipment', 'distance', 'weight', 'market_index', 'quote_signal']])

    # Let's inspect how market_index and quote_signal are generated in validation
    daily_stats = val_dec.groupby('date').agg({
        'market_index': ['count', 'mean', 'std'],
        'quote_signal': ['count', 'mean', 'std']
    })
    print("\nDaily stats in Dec validation (first 10 days):\n", daily_stats.head(10))

if __name__ == "__main__":
    check_december()
