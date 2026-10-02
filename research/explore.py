import pandas as pd
import numpy as np

def explore():
    df_train = pd.read_csv("train-test.csv")
    df_val = pd.read_csv("validation.csv")
    df_dec = pd.read_csv("december-chart-inputs.csv")
    
    print("=== TRAIN-TEST SHAPE & INFO ===")
    print("Shape:", df_train.shape)
    print(df_train.info())
    print("\nMissing values in train-test:\n", df_train.isnull().sum())
    
    print("\n=== VALIDATION SHAPE & INFO ===")
    print("Shape:", df_val.shape)
    print(df_val.info())
    print("\nMissing values in validation:\n", df_val.isnull().sum())
    
    print("\n=== DECEMBER CHART INPUTS ===")
    print("Shape:", df_dec.shape)
    print(df_dec.head(5))
    print(df_dec.isnull().sum())
    
    print("\n=== DATE RANGES ===")
    print("Train dates:", df_train['date'].min(), "to", df_train['date'].max())
    print("Validation dates:", df_val['date'].min(), "to", df_val['date'].max())
    print("December dates:", df_dec['date'].min(), "to", df_dec['date'].max())
    
    print("\n=== SUMMARY STATS TRAIN ===")
    print(df_train.describe().T)
    
    print("\n=== SUMMARY STATS VAL ===")
    print(df_val.describe().T)
    
    print("\n=== CATEGORICAL UNIQUE COUNTS ===")
    for col in ['pickup', 'delivery', 'equipment']:
        print(f"Col {col}: Train unique={df_train[col].nunique()}, Val unique={df_val[col].nunique()}")
        train_cats = set(df_train[col].dropna().unique())
        val_cats = set(df_val[col].dropna().unique())
        print(f"  Val in train? {val_cats.issubset(train_cats)} (Unseen: {val_cats - train_cats})")

    print("\n=== EQUIPMENT VALUE COUNTS ===")
    print("Train equipment:\n", df_train['equipment'].value_counts(dropna=False))
    print("Val equipment:\n", df_val['equipment'].value_counts(dropna=False))

    print("\n=== TARGET (POSTED_RATE) STATS ===")
    print(df_train['posted_rate'].describe())
    print("Rate per mile describe:")
    print((df_train['posted_rate'] / df_train['distance']).describe())

if __name__ == "__main__":
    explore()
