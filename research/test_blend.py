import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from benchmark_models import engineer_features

def test_blend():
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
    
    # Model A: LGBM L1 on RPM
    mA = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    mA.fit(X_train, y_train_rpm)
    pA = mA.predict(X_val) * dist_val
    
    # Model B: LGBM L2 on RPM
    mB = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    mB.fit(X_train, y_train_rpm)
    pB = mB.predict(X_val) * dist_val
    
    # Model C: LGBM L1 Direct
    mC = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    mC.fit(X_train, y_train)
    pC = mC.predict(X_val)

    for w_A in [0.5, 0.6, 0.7, 0.8, 1.0]:
        w_rest = (1.0 - w_A) / 2.0
        p_blend = w_A * pA + w_rest * pB + w_rest * pC
        print(f"Weight A={w_A:.2f}, rest={w_rest:.2f} -> RMSE: {np.sqrt(mean_squared_error(y_val, p_blend)):.2f}, MAE: {mean_absolute_error(y_val, p_blend):.2f}, Median: {np.median(np.abs(y_val - p_blend)):.2f}, R2: {r2_score(y_val, p_blend):.4f}")

if __name__ == "__main__":
    test_blend()
