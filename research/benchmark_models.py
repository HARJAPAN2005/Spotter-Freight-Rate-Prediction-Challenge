import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def haversine_np(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def engineer_features(df):
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    
    # Missing indicators & imputation
    d['weight_isna'] = d['weight'].isna().astype(int)
    d['weight_filled'] = d['weight'].fillna(31000.0)
    
    if 'market_index' in d.columns:
        d['market_index_isna'] = d['market_index'].isna().astype(int)
        d['market_index_filled'] = d['market_index'].fillna(1.0)
    
    # Temporal features
    d['month'] = d['date'].dt.month
    d['day'] = d['date'].dt.day
    d['dayofweek'] = d['date'].dt.dayofweek
    d['dayofyear'] = d['date'].dt.dayofyear
    d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
    d['is_month_end'] = (d['day'] >= 25).astype(int)
    d['quarter'] = d['date'].dt.quarter
    
    d['sin_dayofweek'] = np.sin(2 * np.pi * d['dayofweek'] / 7)
    d['cos_dayofweek'] = np.cos(2 * np.pi * d['dayofweek'] / 7)
    d['sin_dayofyear'] = np.sin(2 * np.pi * d['dayofyear'] / 365.25)
    d['cos_dayofyear'] = np.cos(2 * np.pi * d['dayofyear'] / 365.25)
    
    # Geospatial features
    if 'pickup_lat' in d.columns:
        d['haversine'] = haversine_np(d['pickup_lat'], d['pickup_lon'], d['delivery_lat'], d['delivery_lon'])
        d['tortuosity'] = d['distance'] / (d['haversine'] + 1e-4)
        d['lat_diff'] = np.abs(d['delivery_lat'] - d['pickup_lat'])
        d['lon_diff'] = np.abs(d['delivery_lon'] - d['pickup_lon'])
        d['mid_lat'] = (d['pickup_lat'] + d['delivery_lat']) / 2.0
        d['mid_lon'] = (d['pickup_lon'] + d['delivery_lon']) / 2.0
        
    # Freight & payload features
    d['weight_tier'] = pd.cut(d['weight_filled'], bins=[-np.inf, 20000, 35000, np.inf], labels=[0, 1, 2]).astype(int)
    d['ton_miles'] = (d['weight_filled'] / 2000.0) * d['distance']
    
    # Quote & market interactions if available
    if 'quote_signal' in d.columns:
        d['est_base'] = d['distance'] * d['quote_signal']
        if 'market_index_filled' in d.columns:
            d['market_adj_base'] = d['est_base'] * d['market_index_filled']
            d['quote_x_market'] = d['quote_signal'] * d['market_index_filled']
            
    # Equipment one-hot / encoding
    eq_dummies = pd.get_dummies(d['equipment'], prefix='eq', drop_first=False)
    for col in eq_dummies.columns:
        d[col] = eq_dummies[col].astype(int)
        
    return d

def run_benchmarks():
    df = pd.read_csv("train-test.csv")
    df_feat = engineer_features(df)
    
    # Temporal Split: Jan-Aug (train) vs Sep-Oct (val)
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
    
    X_val = df_feat.loc[val_mask, feature_cols]
    y_val = df_feat.loc[val_mask, 'posted_rate']
    dist_val = df_feat.loc[val_mask, 'distance']
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    results = {}
    
    # 1. LightGBM direct posted_rate
    print("\nTraining LightGBM direct posted_rate...")
    lgb_direct = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    lgb_direct.fit(X_train, y_train)
    p_lgb_dir = lgb_direct.predict(X_val)
    results['LGBM_Direct'] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, p_lgb_dir)),
        'MAE': mean_absolute_error(y_val, p_lgb_dir),
        'R2': r2_score(y_val, p_lgb_dir)
    }
    
    # 2. LightGBM Rate Per Mile (rpm)
    print("Training LightGBM Rate Per Mile...")
    y_train_rpm = y_train / df_feat.loc[train_mask, 'distance']
    lgb_rpm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    lgb_rpm.fit(X_train, y_train_rpm)
    p_lgb_rpm = lgb_rpm.predict(X_val) * dist_val
    results['LGBM_RPM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, p_lgb_rpm)),
        'MAE': mean_absolute_error(y_val, p_lgb_rpm),
        'R2': r2_score(y_val, p_lgb_rpm)
    }

    # 3. XGBoost direct posted_rate
    print("Training XGBoost direct posted_rate...")
    xgb_direct = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)
    xgb_direct.fit(X_train, y_train)
    p_xgb_dir = xgb_direct.predict(X_val)
    results['XGB_Direct'] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, p_xgb_dir)),
        'MAE': mean_absolute_error(y_val, p_xgb_dir),
        'R2': r2_score(y_val, p_xgb_dir)
    }

    # 4. XGBoost Rate Per Mile
    print("Training XGBoost Rate Per Mile...")
    xgb_rpm = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)
    xgb_rpm.fit(X_train, y_train_rpm)
    p_xgb_rpm = xgb_rpm.predict(X_val) * dist_val
    results['XGB_RPM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, p_xgb_rpm)),
        'MAE': mean_absolute_error(y_val, p_xgb_rpm),
        'R2': r2_score(y_val, p_xgb_rpm)
    }

    # 5. Ensemble: Blend of LGBM_RPM and XGB_RPM
    p_ensemble = 0.5 * p_lgb_rpm + 0.5 * p_xgb_rpm
    results['Ensemble_RPM_LGB_XGB'] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, p_ensemble)),
        'MAE': mean_absolute_error(y_val, p_ensemble),
        'R2': r2_score(y_val, p_ensemble)
    }

    # 6. Ensemble: Blend of all four
    p_quad = 0.3 * p_lgb_rpm + 0.3 * p_xgb_rpm + 0.2 * p_lgb_dir + 0.2 * p_xgb_dir
    results['Ensemble_Quad'] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, p_quad)),
        'MAE': mean_absolute_error(y_val, p_quad),
        'R2': r2_score(y_val, p_quad)
    }
    
    print("\n=== BENCHMARK RESULTS (Out-of-Time Sep-Oct Validation) ===")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())

if __name__ == "__main__":
    run_benchmarks()
