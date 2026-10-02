import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

def inspect_models():
    df = pd.read_csv("train-test.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Check date split: e.g. Jan-Aug train, Sep-Oct validation (time-based split)
    # or k-fold CV
    print("Dataset rows:", len(df))
    print("Null counts:\n", df.isnull().sum())
    
    # Feature engineering test
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['est_base'] = df['distance'] * df['quote_signal']
    
    # Impute missing weight and market_index
    # What are the missing values?
    print("Missing weight by equipment:")
    print(df[df['weight'].isna()]['equipment'].value_counts())
    print("Overall weight mean by equipment:\n", df.groupby('equipment')['weight'].mean())
    
    # Check market_index missing values
    # Is market_index missing on specific dates, or random?
    print("Missing market index dates:", df[df['market_index'].isna()]['date'].nunique())
    
    # Let's inspect rows with extreme posted_rate
    df['rpm'] = df['posted_rate'] / df['distance']
    print("\nTop 10 highest rpm:")
    print(df.sort_values('rpm', ascending=False)[['date', 'equipment', 'distance', 'weight', 'quote_signal', 'market_index', 'posted_rate', 'rpm']].head(10))

    print("\nTop 10 lowest rpm:")
    print(df.sort_values('rpm', ascending=True)[['date', 'equipment', 'distance', 'weight', 'quote_signal', 'market_index', 'posted_rate', 'rpm']].head(10))

if __name__ == "__main__":
    inspect_models()
