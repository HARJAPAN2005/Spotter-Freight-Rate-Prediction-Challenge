import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_residuals():
    df = pd.read_csv("train-test.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    train_mask = df['date'] < '2025-09-01'
    val_mask = df['date'] >= '2025-09-01'
    
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Let's inspect city encoding / lane encoding
    df['lane'] = df['pickup'] + " -> " + df['delivery']
    
    features = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'weight', 'month', 'day', 'dayofweek',
        'dayofyear', 'is_weekend'
    ]
    # One-hot equipment
    eq_dummies = pd.get_dummies(df['equipment'], prefix='eq', drop_first=True)
    df_feat = pd.concat([df[features], eq_dummies], axis=1)
    
    target = df['posted_rate']
    
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    m.fit(df_feat.loc[train_mask], target.loc[train_mask])
    
    preds = m.predict(df_feat.loc[val_mask])
    actual = target.loc[val_mask]
    
    errors = actual - preds
    abs_errors = np.abs(errors)
    
    print("Residual summary:")
    print("Mean error:", errors.mean())
    print("Median abs error:", np.median(abs_errors))
    print("90th percentile abs error:", np.percentile(abs_errors, 90))
    print("95th percentile abs error:", np.percentile(abs_errors, 95))
    print("99th percentile abs error:", np.percentile(abs_errors, 99))
    print("Max abs error:", abs_errors.max())
    
    # Look at largest errors
    df_val = df.loc[val_mask].copy()
    df_val['pred'] = preds
    df_val['error'] = errors
    df_val['abs_error'] = abs_errors
    
    print("\nTop 10 largest errors:")
    print(df_val.sort_values('abs_error', ascending=False)[['date', 'lane', 'distance', 'equipment', 'weight', 'posted_rate', 'pred', 'error']].head(10))

if __name__ == "__main__":
    analyze_residuals()
