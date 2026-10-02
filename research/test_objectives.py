import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from benchmark_models import engineer_features

def test_objectives():
    df = pd.read_csv("train-test.csv")
    df_feat = engineer_features(df)
    
    train_mask = df_feat['date'] < '2025-09-01'
    val_mask = df_feat['date'] >= '2025-09-01'
    
    feature_cols = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'haversine', 'tortuosity', 'lat_diff', 'lon_diff', 'mid_lat', 'mid_lon',
        'distance', 'weight_filled', 'weight_isna', 'weight_tier', 'ton_miles',
        'month', 'day', 'dayofweek', 'dayofyear', 'is_weekend', 'is_month_end', 'quarter',
        'sin_dayofweek', 'cos_dayofweek', 'sin_dayofyear', 'cos_dayofyear',
        'eq_Dry Van', 'eq_Flatbed', 'eq_Reefer',
        'quote_signal', 'est_base', 'market_index_filled', 'market_index_isna',
        'market_adj_base', 'quote_x_market'
    ]
    
    X_train = df_feat.loc[train_mask, feature_cols]
    y_train = df_feat.loc[train_mask, 'posted_rate']
    y_train_rpm = y_train / df_feat.loc[train_mask, 'distance']
    
    X_val = df_feat.loc[val_mask, feature_cols]
    y_val = df_feat.loc[val_mask, 'posted_rate']
    dist_val = df_feat.loc[val_mask, 'distance']
    
    objectives = ['regression', 'regression_l1', 'huber']
    
    for obj in objectives:
        # RPM
        m_rpm = lgb.LGBMRegressor(objective=obj, n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
        m_rpm.fit(X_train, y_train_rpm)
        p_rpm = m_rpm.predict(X_val) * dist_val
        print(f"\n--- Objective: {obj} (Target: RPM) ---")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_val, p_rpm)):.2f}")
        print(f"MAE:  {mean_absolute_error(y_val, p_rpm):.2f}")
        print(f"Median Abs Err: {np.median(np.abs(y_val - p_rpm)):.2f}")
        print(f"R2:   {r2_score(y_val, p_rpm):.4f}")

        # Direct
        m_dir = lgb.LGBMRegressor(objective=obj, n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
        m_dir.fit(X_train, y_train)
        p_dir = m_dir.predict(X_val)
        print(f"--- Objective: {obj} (Target: Direct) ---")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_val, p_dir)):.2f}")
        print(f"MAE:  {mean_absolute_error(y_val, p_dir):.2f}")
        print(f"Median Abs Err: {np.median(np.abs(y_val - p_dir)):.2f}")
        print(f"R2:   {r2_score(y_val, p_dir):.4f}")

if __name__ == "__main__":
    test_objectives()
