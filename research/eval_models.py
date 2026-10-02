import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models():
    df = pd.read_csv("train-test.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date for temporal validation
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split: Train on Jan-Aug (months 1-8), Validate on Sep-Oct (months 9-10)
    train_mask = df['date'] < '2025-09-01'
    val_mask = df['date'] >= '2025-09-01'
    
    print(f"Train rows (Jan-Aug): {train_mask.sum()}")
    print(f"Val rows (Sep-Oct): {val_mask.sum()}")
    
    # Feature engineering
    for d in [df]:
        d['month'] = d['date'].dt.month
        d['day'] = d['date'].dt.day
        d['dayofweek'] = d['date'].dt.dayofweek
        d['dayofyear'] = d['date'].dt.dayofyear
        d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
        d['equipment'] = d['equipment'].astype('category')
        d['pickup'] = d['pickup'].astype('category')
        d['delivery'] = d['delivery'].astype('category')
        
    features_full = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'equipment', 'weight', 'month', 'day', 'dayofweek',
        'dayofyear', 'is_weekend', 'market_index', 'quote_signal'
    ]
    
    features_no_signals = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'equipment', 'weight', 'month', 'day', 'dayofweek',
        'dayofyear', 'is_weekend'
    ]
    
    target = 'posted_rate'
    
    # Model 1: LightGBM with all features
    print("\n--- Model 1: LightGBM with quote_signal and market_index ---")
    m1 = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    m1.fit(df.loc[train_mask, features_full], df.loc[train_mask, target])
    preds1 = m1.predict(df.loc[val_mask, features_full])
    y_val = df.loc[val_mask, target]
    print(f"RMSE: {mean_squared_error(y_val, preds1)**0.5:.2f}")
    print(f"MAE:  {mean_absolute_error(y_val, preds1):.2f}")
    print(f"R2:   {r2_score(y_val, preds1):.4f}")
    
    # Model 2: LightGBM without quote_signal and market_index
    print("\n--- Model 2: LightGBM WITHOUT quote_signal and market_index ---")
    m2 = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    m2.fit(df.loc[train_mask, features_no_signals], df.loc[train_mask, target])
    preds2 = m2.predict(df.loc[val_mask, features_no_signals])
    print(f"RMSE: {mean_squared_error(y_val, preds2)**0.5:.2f}")
    print(f"MAE:  {mean_absolute_error(y_val, preds2):.2f}")
    print(f"R2:   {r2_score(y_val, preds2):.4f}")

    # Model 3: Predicting rate per mile vs predicting posted_rate
    print("\n--- Model 3: LightGBM predicting rate per mile (with quote_signal & market_index) ---")
    y_rpm_train = df.loc[train_mask, target] / df.loc[train_mask, 'distance']
    m3 = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    m3.fit(df.loc[train_mask, features_full], y_rpm_train)
    preds_rpm = m3.predict(df.loc[val_mask, features_full])
    preds3 = preds_rpm * df.loc[val_mask, 'distance']
    print(f"RMSE: {mean_squared_error(y_val, preds3)**0.5:.2f}")
    print(f"MAE:  {mean_absolute_error(y_val, preds3):.2f}")
    print(f"R2:   {r2_score(y_val, preds3):.4f}")

    # Model 4: Predicting ratio to (distance * quote_signal)
    print("\n--- Model 4: LightGBM predicting ratio = posted_rate / (distance * quote_signal) ---")
    base_train = df.loc[train_mask, 'distance'] * df.loc[train_mask, 'quote_signal']
    y_ratio_train = df.loc[train_mask, target] / base_train
    m4 = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    m4.fit(df.loc[train_mask, features_full], y_ratio_train)
    preds_ratio = m4.predict(df.loc[val_mask, features_full])
    base_val = df.loc[val_mask, 'distance'] * df.loc[val_mask, 'quote_signal']
    preds4 = preds_ratio * base_val
    print(f"RMSE: {mean_squared_error(y_val, preds4)**0.5:.2f}")
    print(f"MAE:  {mean_absolute_error(y_val, preds4):.2f}")
    print(f"R2:   {r2_score(y_val, preds4):.4f}")

if __name__ == "__main__":
    evaluate_models()
