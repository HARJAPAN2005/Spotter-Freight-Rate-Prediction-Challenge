"""
Freight Rate Prediction Challenge - Production Training & Prediction Pipeline
Author: Candidate
Task: Predict freight posted rates for validation loads and fixed December lane.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate Great-Circle distance between two points on Earth in miles."""
    R = 3958.8  # Earth radius in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_city_coord_map(train_df, val_df):
    """Build a lookup dictionary of geographic coordinates for all known cities."""
    combined = pd.concat([
        train_df[['pickup', 'pickup_lat', 'pickup_lon']].rename(
            columns={'pickup': 'city', 'pickup_lat': 'lat', 'pickup_lon': 'lon'}
        ),
        val_df[['pickup', 'pickup_lat', 'pickup_lon']].rename(
            columns={'pickup': 'city', 'pickup_lat': 'lat', 'pickup_lon': 'lon'}
        ),
        train_df[['delivery', 'delivery_lat', 'delivery_lon']].rename(
            columns={'delivery': 'city', 'delivery_lat': 'lat', 'delivery_lon': 'lon'}
        ),
        val_df[['delivery', 'delivery_lat', 'delivery_lon']].rename(
            columns={'delivery': 'city', 'delivery_lat': 'lat', 'delivery_lon': 'lon'}
        )
    ]).drop_duplicates(subset=['city']).set_index('city')
    return combined.to_dict('index')


def engineer_features(df, city_coords=None, daily_market_index_map=None, default_qs_map=None):
    """
    Comprehensive feature engineering for freight rate prediction:
    - Temporal & Calendar Dynamics (day of week, cyclical day of year, weekend, month end)
    - Geospatial & Lane Routing (Haversine distance, tortuosity ratio, coordinate deltas)
    - Equipment & Payload Characteristics (weight tiers, ton-miles)
    - Market & Broker Signals (quote signal, market index, interaction baselines)
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    
    # 1. Fill missing coordinates if needed (e.g. for December chart inputs)
    if 'pickup_lat' not in d.columns or d['pickup_lat'].isna().any():
        if city_coords is not None:
            if 'pickup_lat' not in d.columns:
                d['pickup_lat'] = d['pickup'].map(lambda c: city_coords.get(c, {}).get('lat', np.nan))
                d['pickup_lon'] = d['pickup'].map(lambda c: city_coords.get(c, {}).get('lon', np.nan))
                d['delivery_lat'] = d['delivery'].map(lambda c: city_coords.get(c, {}).get('lat', np.nan))
                d['delivery_lon'] = d['delivery'].map(lambda c: city_coords.get(c, {}).get('lon', np.nan))
            else:
                d['pickup_lat'] = d['pickup_lat'].fillna(d['pickup'].map(lambda c: city_coords.get(c, {}).get('lat', np.nan)))
                d['pickup_lon'] = d['pickup_lon'].fillna(d['pickup'].map(lambda c: city_coords.get(c, {}).get('lon', np.nan)))
                d['delivery_lat'] = d['delivery_lat'].fillna(d['delivery'].map(lambda c: city_coords.get(c, {}).get('lat', np.nan)))
                d['delivery_lon'] = d['delivery_lon'].fillna(d['delivery'].map(lambda c: city_coords.get(c, {}).get('lon', np.nan)))

    # 2. Market Index handling
    if 'market_index' not in d.columns:
        if daily_market_index_map is not None:
            d['market_index'] = d['date'].dt.date.map(daily_market_index_map)
        else:
            d['market_index'] = 1.0
            
    d['market_index_isna'] = d['market_index'].isna().astype(int)
    # If market_index is missing, fill with daily average if available, else 1.0
    if daily_market_index_map is not None:
        d['market_index_filled'] = d['market_index'].fillna(d['date'].dt.date.map(daily_market_index_map)).fillna(1.0)
    else:
        d['market_index_filled'] = d['market_index'].fillna(1.0)
        
    # 3. Quote Signal handling
    if 'quote_signal' not in d.columns:
        lane_key = d['pickup'] + " -> " + d['delivery'] + " | " + d['equipment']
        if default_qs_map is not None:
            d['quote_signal'] = lane_key.map(default_qs_map).fillna(2.05)
        else:
            d['quote_signal'] = 2.05
    else:
        d['quote_signal'] = d['quote_signal'].fillna(2.05)
        
    # 4. Payload & Weight
    d['weight_isna'] = d['weight'].isna().astype(int)
    d['weight_filled'] = d['weight'].fillna(31000.0)
    d['weight_tier'] = pd.cut(
        d['weight_filled'],
        bins=[-np.inf, 20000, 35000, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    d['ton_miles'] = (d['weight_filled'] / 2000.0) * d['distance']
    
    # 5. Temporal features
    d['month'] = d['date'].dt.month
    d['day'] = d['date'].dt.day
    d['dayofweek'] = d['date'].dt.dayofweek
    d['dayofyear'] = d['date'].dt.dayofyear
    d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
    d['is_month_end'] = (d['day'] >= 25).astype(int)
    d['quarter'] = d['date'].dt.quarter
    
    d['sin_dayofweek'] = np.sin(2 * np.pi * d['dayofweek'] / 7.0)
    d['cos_dayofweek'] = np.cos(2 * np.pi * d['dayofweek'] / 7.0)
    d['sin_dayofyear'] = np.sin(2 * np.pi * d['dayofyear'] / 365.25)
    d['cos_dayofyear'] = np.cos(2 * np.pi * d['dayofyear'] / 365.25)

    # 6. Geospatial & Routing
    d['haversine'] = haversine_np(d['pickup_lat'], d['pickup_lon'], d['delivery_lat'], d['delivery_lon'])
    d['tortuosity'] = d['distance'] / (d['haversine'] + 1e-4)
    d['lat_diff'] = np.abs(d['delivery_lat'] - d['pickup_lat'])
    d['lon_diff'] = np.abs(d['delivery_lon'] - d['pickup_lon'])
    d['mid_lat'] = (d['pickup_lat'] + d['delivery_lat']) / 2.0
    d['mid_lon'] = (d['pickup_lon'] + d['delivery_lon']) / 2.0
    
    # 7. Quote & Market interactions
    d['est_base'] = d['distance'] * d['quote_signal']
    d['market_adj_base'] = d['est_base'] * d['market_index_filled']
    d['quote_x_market'] = d['quote_signal'] * d['market_index_filled']
    
    # 8. Equipment one-hot encoding
    for eq in ['Dry Van', 'Flatbed', 'Reefer']:
        d[f'eq_{eq}'] = (d['equipment'] == eq).astype(int)

    return d


def main():
    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - ML TRAINING & INFERENCE PIPELINE")
    print("=" * 70)
    
    # Locate files
    train_path = Path("train-test.csv") if Path("train-test.csv").is_file() else Path("data/train_test.csv")
    val_path = Path("validation.csv") if Path("validation.csv").is_file() else Path("data/validation.csv")
    dec_path = Path("december-chart-inputs.csv") if Path("december-chart-inputs.csv").is_file() else Path("data/december_chart_inputs.csv")
    
    print(f"Loading data from:")
    print(f"  Train:      {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  December:   {dec_path}")
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_dec = pd.read_csv(dec_path)
    
    # Build reference lookup tables
    city_coords = build_city_coord_map(df_train, df_val)
    print(f"Constructed geographic coordinates lookup for {len(city_coords)} unique cities.")
    
    # Compute daily market_index lookup from train + validation
    df_all_dates = pd.concat([
        df_train[['date', 'market_index']].dropna(),
        df_val[['date', 'market_index']].dropna()
    ])
    df_all_dates['date'] = pd.to_datetime(df_all_dates['date']).dt.date
    daily_market_index_map = df_all_dates.groupby('date')['market_index'].mean().to_dict()
    print(f"Constructed daily market_index lookup across {len(daily_market_index_map)} active dates.")
    
    # Compute lane/equipment quote signal lookup
    df_train_lanes = df_train.copy()
    df_train_lanes['lane_key'] = df_train_lanes['pickup'] + " -> " + df_train_lanes['delivery'] + " | " + df_train_lanes['equipment']
    df_val_lanes = df_val.copy()
    df_val_lanes['lane_key'] = df_val_lanes['pickup'] + " -> " + df_val_lanes['delivery'] + " | " + df_val_lanes['equipment']
    
    lane_qs_df = pd.concat([
        df_train_lanes[['lane_key', 'quote_signal']].dropna(),
        df_val_lanes[['lane_key', 'quote_signal']].dropna()
    ])
    lane_qs_map = lane_qs_df.groupby('lane_key')['quote_signal'].mean().to_dict()
    print(f"Constructed lane quote_signal lookup for {len(lane_qs_map)} lanes.")

    # Feature Engineering
    print("\nEngineering features...")
    df_train_feat = engineer_features(df_train, city_coords, daily_market_index_map, lane_qs_map)
    df_val_feat = engineer_features(df_val, city_coords, daily_market_index_map, lane_qs_map)
    df_dec_feat = engineer_features(df_dec, city_coords, daily_market_index_map, lane_qs_map)

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
    
    print(f"Total features per record: {len(feature_cols)}")
    
    # -------------------------------------------------------------
    # 1. OUT-OF-TIME (OOT) TEMPORAL VALIDATION EXPERIMENT
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 1: OUT-OF-TIME VALIDATION EXPERIMENT (Jan-Aug -> Sep-Oct)")
    print("=" * 50)
    
    oot_train_mask = df_train_feat['date'] < '2025-09-01'
    oot_val_mask = df_train_feat['date'] >= '2025-09-01'
    
    X_oot_train = df_train_feat.loc[oot_train_mask, feature_cols]
    y_oot_train = df_train_feat.loc[oot_train_mask, 'posted_rate']
    y_oot_train_rpm = y_oot_train / df_train_feat.loc[oot_train_mask, 'distance']
    
    X_oot_val = df_train_feat.loc[oot_val_mask, feature_cols]
    y_oot_val = df_train_feat.loc[oot_val_mask, 'posted_rate']
    dist_oot_val = df_train_feat.loc[oot_val_mask, 'distance']
    
    print(f"OOT Train samples: {len(X_oot_train):,} (Jan 01, 2025 - Aug 31, 2025)")
    print(f"OOT Validation samples: {len(X_oot_val):,} (Sep 01, 2025 - Oct 31, 2025)")
    
    # Train OOT ensemble members
    print("Fitting OOT Model 1: LightGBM L1 (Rate Per Mile)...")
    m1_oot = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    m1_oot.fit(X_oot_train, y_oot_train_rpm)
    p1_oot = m1_oot.predict(X_oot_val) * dist_oot_val
    
    print("Fitting OOT Model 2: LightGBM L2 (Rate Per Mile)...")
    m2_oot = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    m2_oot.fit(X_oot_train, y_oot_train_rpm)
    p2_oot = m2_oot.predict(X_oot_val) * dist_oot_val
    
    print("Fitting OOT Model 3: LightGBM L1 (Direct Posted Rate)...")
    m3_oot = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    m3_oot.fit(X_oot_train, y_oot_train)
    p3_oot = m3_oot.predict(X_oot_val)
    
    p_oot_ensemble = 0.70 * p1_oot + 0.15 * p2_oot + 0.15 * p3_oot
    
    rmse = np.sqrt(mean_squared_error(y_oot_val, p_oot_ensemble))
    mae = mean_absolute_error(y_oot_val, p_oot_ensemble)
    med_ae = np.median(np.abs(y_oot_val - p_oot_ensemble))
    r2 = r2_score(y_oot_val, p_oot_ensemble)
    mape = np.mean(np.abs((y_oot_val - p_oot_ensemble) / y_oot_val)) * 100.0
    
    print("\n--- OOT VALIDATION RESULTS (Out-of-Time Sep-Oct 2025) ---")
    print(f"  RMSE:               ${rmse:.2f}")
    print(f"  MAE:                ${mae:.2f}")
    print(f"  Median Abs Error:   ${med_ae:.2f}")
    print(f"  R-Squared (R2):     {r2:.4f}")
    print(f"  MAPE:               {mape:.2f}%")
    
    # -------------------------------------------------------------
    # 2. FULL DEVELOPMENT TRAINING FOR INFERENCE
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 2: FULL TRAINING ON ALL DEVELOPMENT DATA (48,000 loads)")
    print("=" * 50)
    
    X_full = df_train_feat[feature_cols]
    y_full = df_train_feat['posted_rate']
    y_full_rpm = y_full / df_train_feat['distance']
    
    print(f"Fitting Final Model 1: LightGBM L1 (RPM) on {len(X_full):,} rows...")
    final_m1 = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1200, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    final_m1.fit(X_full, y_full_rpm)
    
    print(f"Fitting Final Model 2: LightGBM L2 (RPM) on {len(X_full):,} rows...")
    final_m2 = lgb.LGBMRegressor(objective='regression', n_estimators=1200, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    final_m2.fit(X_full, y_full_rpm)
    
    print(f"Fitting Final Model 3: LightGBM L1 (Direct) on {len(X_full):,} rows...")
    final_m3 = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1200, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    final_m3.fit(X_full, y_full)

    # -------------------------------------------------------------
    # 3. GENERATE PREDICTIONS FOR VALIDATION.CSV (12,000 loads)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 3: PREDICTING VALIDATION SET (12,000 LOADS)")
    print("=" * 50)
    
    X_val = df_val_feat[feature_cols]
    dist_val = df_val_feat['distance']
    
    pred_val_1 = final_m1.predict(X_val) * dist_val
    pred_val_2 = final_m2.predict(X_val) * dist_val
    pred_val_3 = final_m3.predict(X_val)
    
    val_final_preds = 0.70 * pred_val_1 + 0.15 * pred_val_2 + 0.15 * pred_val_3
    # Ensure all predictions are positive
    val_final_preds = np.clip(val_final_preds, 50.0, None)
    
    sub_df = pd.DataFrame({
        'load_id': df_val['load_id'],
        'predicted_rate': np.round(val_final_preds, 2)
    })
    
    out_val_csv = Path("validation_predictions.csv")
    sub_df.to_csv(out_val_csv, index=False)
    print(f"Saved {len(sub_df):,} predictions to: {out_val_csv.resolve()}")
    print("Sample validation predictions:")
    print(sub_df.head(5))

    # -------------------------------------------------------------
    # 4. GENERATE PREDICTIONS FOR DECEMBER CHART (31 days)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 4: PREDICTING FIXED DECEMBER CHART (31 DAYS)")
    print("=" * 50)
    
    X_dec = df_dec_feat[feature_cols]
    dist_dec = df_dec_feat['distance']
    
    pred_dec_1 = final_m1.predict(X_dec) * dist_dec
    pred_dec_2 = final_m2.predict(X_dec) * dist_dec
    pred_dec_3 = final_m3.predict(X_dec)
    
    dec_final_preds = 0.70 * pred_dec_1 + 0.15 * pred_dec_2 + 0.15 * pred_dec_3
    dec_final_preds = np.clip(dec_final_preds, 50.0, None)
    
    # Save to both root and data/ for compatibility with all run configurations
    df_dec_result = df_dec.copy()
    df_dec_result['predicted_rate'] = np.round(dec_final_preds, 2)
    
    out_dec_csv1 = Path("december-chart-inputs.csv")
    out_dec_csv2 = Path("data/december_chart_inputs.csv")
    out_dec_csv3 = Path("data/december-chart-inputs.csv")
    
    df_dec_result.to_csv(out_dec_csv1, index=False)
    df_dec_result.to_csv(out_dec_csv2, index=False)
    df_dec_result.to_csv(out_dec_csv3, index=False)
    
    print(f"Saved December predictions to: {out_dec_csv1} and {out_dec_csv2}")
    print("December predictions preview (first 7 days):")
    print(df_dec_result.head(7))
    print(f"December rate summary: Min=${dec_final_preds.min():.2f}, Mean=${dec_final_preds.mean():.2f}, Max=${dec_final_preds.max():.2f}")
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
