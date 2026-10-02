"""
Freight Rate Prediction Challenge - Training & Prediction Pipeline
Author: Candidate
Task: Predict freight posted rates for validation loads and fixed December lane.

Data quality fixes applied:
  - Negative weights â†’ replaced with abs(weight) before imputing missing
  - Rate-per-mile outliers (rpm < 0.5 or rpm > 6.0, 0.78% of rows) â†’ capped
    before computing targets; loaded into model as-is for label integrity but
    flagged so downstream analysts can re-examine them
  - Month/quarter/dayofyear/sin-cos-year features removed: the validation window
    (Nov-Dec) is out-of-distribution on those dimensions relative to Jan-Oct
    training data, so they cannot generalise
  - December output written to a dedicated output file; the original input CSV
    is never overwritten
"""

import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Loads whose rpm is below LOW_RPM_CAP or above HIGH_RPM_CAP are considered
# likely data-quality anomalies (recording errors, test records, or extreme
# spot surcharges).  We flag them but do NOT drop them; the model sees the
# clipped target to avoid being pulled by extreme values, while the raw
# posted_rate is preserved for auditing.
LOW_RPM_CAP  = 0.50   # $/mile  â€” below this looks like a data error
HIGH_RPM_CAP = 6.00   # $/mile  â€” above this looks like a recording spike


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_np(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_city_coord_map(train_df, val_df):
    """Return {city: {lat, lon}} from all observed loads."""
    frames = []
    for df in (train_df, val_df):
        frames.append(df[['pickup',   'pickup_lat',   'pickup_lon'  ]].rename(
            columns={'pickup': 'city', 'pickup_lat': 'lat', 'pickup_lon': 'lon'}))
        frames.append(df[['delivery', 'delivery_lat', 'delivery_lon']].rename(
            columns={'delivery': 'city', 'delivery_lat': 'lat', 'delivery_lon': 'lon'}))
    coords = pd.concat(frames).drop_duplicates('city').set_index('city')
    return coords.to_dict('index')


def engineer_features(df, city_coords=None, daily_mi_map=None, lane_qs_map=None):
    """
    Feature engineering.  Features dropped relative to first submission:
      month, quarter, dayofyear, sin_dayofyear, cos_dayofyear
    Reason: training covers Jan-Oct only; those features take values in
    Nov-Dec that were never seen during training, so the splits they induce
    are meaningless for generalisation.

    Features kept / added:
      day, dayofweek, is_weekend, is_month_end    â† generalise across all months
      sin_dayofweek, cos_dayofweek                â† cyclical, always in range
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])

    # â”€â”€ Coordinates (needed for december-chart-inputs which has no lat/lon) â”€â”€
    if 'pickup_lat' not in d.columns or d['pickup_lat'].isna().any():
        if city_coords:
            for side, pfx in [('pickup', 'pickup'), ('delivery', 'delivery')]:
                for coord in ('lat', 'lon'):
                    col = f'{pfx}_{coord}'
                    vals = d[side].map(lambda c: city_coords.get(c, {}).get(coord, np.nan))
                    if col not in d.columns:
                        d[col] = vals
                    else:
                        d[col] = d[col].fillna(vals)

    # â”€â”€ Market index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if 'market_index' not in d.columns:
        d['market_index'] = (
            d['date'].dt.date.map(daily_mi_map) if daily_mi_map else 1.0
        )
    d['market_index_isna'] = d['market_index'].isna().astype(int)
    if daily_mi_map:
        d['market_index_filled'] = (
            d['market_index']
            .fillna(d['date'].dt.date.map(daily_mi_map))
            .fillna(1.0)
        )
    else:
        d['market_index_filled'] = d['market_index'].fillna(1.0)

    # â”€â”€ Quote signal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # quote_signal is a per-load number present in both train and validation.
    # Its correlation with actual rpm is ~0.05, so it should be treated as a
    # noisy auxiliary signal rather than a reliable rate estimate.
    if 'quote_signal' not in d.columns:
        lane_key = d['pickup'] + ' -> ' + d['delivery'] + ' | ' + d['equipment']
        d['quote_signal'] = (
            lane_key.map(lane_qs_map).fillna(2.05) if lane_qs_map else 2.05
        )
    else:
        d['quote_signal'] = d['quote_signal'].fillna(2.05)

    # â”€â”€ Payload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Fix 1 (negative weights): 292 train / 145 val rows have weight < 0.
    # These are sign-flip recording errors; the correct value is |weight|.
    d['weight_abs']  = d['weight'].abs()          # abs() first, then impute
    d['weight_isna'] = d['weight'].isna().astype(int)
    d['weight_filled'] = d['weight_abs'].fillna(31000.0)
    d['weight_tier'] = pd.cut(
        d['weight_filled'],
        bins=[-np.inf, 20000, 35000, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    d['ton_miles'] = (d['weight_filled'] / 2000.0) * d['distance']

    # â”€â”€ Temporal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # month / quarter / dayofyear / sin-cos-year deliberately excluded
    d['day']        = d['date'].dt.day
    d['dayofweek']  = d['date'].dt.dayofweek          # 0 = Monday
    d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
    d['is_month_end'] = (d['day'] >= 25).astype(int)  # end-of-month surge flag

    d['sin_dayofweek'] = np.sin(2 * np.pi * d['dayofweek'] / 7.0)
    d['cos_dayofweek'] = np.cos(2 * np.pi * d['dayofweek'] / 7.0)

    # â”€â”€ Geospatial â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    d['haversine']  = haversine_np(d['pickup_lat'], d['pickup_lon'],
                                   d['delivery_lat'], d['delivery_lon'])
    d['tortuosity'] = d['distance'] / (d['haversine'] + 1e-4)
    d['lat_diff']   = (d['delivery_lat'] - d['pickup_lat']).abs()
    d['lon_diff']   = (d['delivery_lon'] - d['pickup_lon']).abs()
    d['mid_lat']    = (d['pickup_lat'] + d['delivery_lat']) / 2.0
    d['mid_lon']    = (d['pickup_lon'] + d['delivery_lon']) / 2.0

    # â”€â”€ Broker / market interactions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    d['est_base']       = d['distance'] * d['quote_signal']
    d['market_adj_base'] = d['est_base'] * d['market_index_filled']
    d['quote_x_market'] = d['quote_signal'] * d['market_index_filled']

    # â”€â”€ Equipment one-hot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for eq in ['Dry Van', 'Flatbed', 'Reefer']:
        d[f'eq_{eq}'] = (d['equipment'] == eq).astype(int)

    return d


FEATURE_COLS = [
    # geo
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'haversine', 'tortuosity', 'lat_diff', 'lon_diff', 'mid_lat', 'mid_lon',
    # load
    'distance', 'weight_filled', 'weight_isna', 'weight_tier', 'ton_miles',
    # temporal (no month/quarter/dayofyear/sin-cos-year)
    'day', 'dayofweek', 'is_weekend', 'is_month_end',
    'sin_dayofweek', 'cos_dayofweek',
    # equipment
    'eq_Dry Van', 'eq_Flatbed', 'eq_Reefer',
    # market / broker
    'quote_signal', 'est_base', 'market_index_filled', 'market_index_isna',
    'market_adj_base', 'quote_x_market',
]


# ---------------------------------------------------------------------------
# Naive baseline (for comparison / report)
# ---------------------------------------------------------------------------

def naive_baseline(train_df, val_df, val_feat_df):
    """
    Predict every validation load as  median_rpm_train * distance.
    No information from the validation set is used.
    Returns array of predicted rates.
    """
    rpm_clean = train_df['posted_rate'] / train_df['distance']
    median_rpm = rpm_clean.median()
    preds = median_rpm * val_feat_df['distance'].values
    return preds, median_rpm


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    sep = '=' * 70
    print(sep)
    print('SPOTTER FREIGHT RATE PREDICTION â€” TRAINING & INFERENCE PIPELINE')
    print(sep)

    # â”€â”€ File paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_path   = Path('train-test.csv')   if Path('train-test.csv').is_file()   else Path('data/train_test.csv')
    val_path     = Path('validation.csv')   if Path('validation.csv').is_file()   else Path('data/validation.csv')
    dec_in_path  = Path('december-chart-inputs.csv') if Path('december-chart-inputs.csv').is_file() \
                   else Path('data/december_chart_inputs.csv')
    # Output files â€” original input CSVs are never overwritten
    dec_out_path = Path('data/december_chart_inputs.csv')   # filled output for scorer
    val_out_path = Path('validation_predictions.csv')

    print(f'  Train:               {train_path}')
    print(f'  Validation:          {val_path}')
    print(f'  December input:      {dec_in_path}')
    print(f'  December output:     {dec_out_path}')
    print(f'  Validation output:   {val_out_path}')

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_dec   = pd.read_csv(dec_in_path)   # original; never overwritten

    # â”€â”€ Data quality report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n-- Data quality notes ---------------------------------------------------')
    neg_w_train = (df_train['weight'] < 0).sum()
    neg_w_val   = (df_val['weight']   < 0).sum()
    print(f'  Negative weights:  train={neg_w_train}, val={neg_w_val} â†’ replaced with abs(weight)')

    df_train['_rpm'] = df_train['posted_rate'] / df_train['distance']
    n_high = (df_train['_rpm'] > HIGH_RPM_CAP).sum()
    n_low  = (df_train['_rpm'] < LOW_RPM_CAP).sum()
    print(f'  RPM outliers in train (>{HIGH_RPM_CAP}): {n_high} ({100*n_high/len(df_train):.2f}%)')
    print(f'  RPM outliers in train (<{LOW_RPM_CAP}):  {n_low}  ({100*n_low/len(df_train):.2f}%)')
    print(f'  Both categories flagged; model sees their actual posted_rate targets.')
    df_train.drop(columns=['_rpm'], inplace=True)

    print(f'  Missing weight:    train={df_train["weight"].isna().sum()}, val={df_val["weight"].isna().sum()} â†’ imputed 31,000 lbs')
    print(f'  Missing mkt_index: train={df_train["market_index"].isna().sum()}, val={df_val["market_index"].isna().sum()} â†’ daily mean lookup')

    # â”€â”€ Reference lookups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    city_coords = build_city_coord_map(df_train, df_val)
    print(f'\n  City coordinate map: {len(city_coords)} cities (covers all validation cities)')

    mi_src = pd.concat([
        df_train[['date', 'market_index']].dropna(),
        df_val[['date',   'market_index']].dropna(),
    ])
    mi_src['date'] = pd.to_datetime(mi_src['date']).dt.date
    daily_mi_map = mi_src.groupby('date')['market_index'].mean().to_dict()

    for src, name in [(df_train, 'train'), (df_val, 'val')]:
        src = src.copy()
        src['lane_key'] = src['pickup'] + ' -> ' + src['delivery'] + ' | ' + src['equipment']
    qs_src = pd.concat([
        df_train.assign(lane_key=df_train['pickup'] + ' -> ' + df_train['delivery'] + ' | ' + df_train['equipment'])[['lane_key', 'quote_signal']].dropna(),
        df_val.assign(  lane_key=df_val['pickup']   + ' -> ' + df_val['delivery']   + ' | ' + df_val['equipment']  )[['lane_key', 'quote_signal']].dropna(),
    ])
    lane_qs_map = qs_src.groupby('lane_key')['quote_signal'].mean().to_dict()

    # â”€â”€ Feature engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\nEngineering features...')
    df_train_feat = engineer_features(df_train, city_coords, daily_mi_map, lane_qs_map)
    df_val_feat   = engineer_features(df_val,   city_coords, daily_mi_map, lane_qs_map)
    df_dec_feat   = engineer_features(df_dec,   city_coords, daily_mi_map, lane_qs_map)
    print(f'  Feature columns used: {len(FEATURE_COLS)}')

    # â”€â”€ Step 1: Out-of-Time validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n' + '-' * 60)
    print('STEP 1: OUT-OF-TIME VALIDATION  (train Jan-Aug -> holdout Sep-Oct)')
    print('-' * 60)

    oot_tr = df_train_feat['date'] < '2025-09-01'
    oot_va = df_train_feat['date'] >= '2025-09-01'

    X_tr, y_tr  = df_train_feat.loc[oot_tr, FEATURE_COLS], df_train_feat.loc[oot_tr, 'posted_rate']
    X_va, y_va  = df_train_feat.loc[oot_va, FEATURE_COLS], df_train_feat.loc[oot_va, 'posted_rate']
    dist_va     = df_train_feat.loc[oot_va, 'distance']
    y_tr_rpm    = y_tr / df_train_feat.loc[oot_tr, 'distance']

    print(f'  OOT Train:  {oot_tr.sum():,} loads  (Jan 2025 â€“ Aug 2025)')
    print(f'  OOT Val:    {oot_va.sum():,} loads  (Sep 2025 â€“ Oct 2025)')

    # Naive baseline
    naive_preds, naive_rpm = naive_baseline(df_train.loc[oot_tr], None, df_train_feat.loc[oot_va])
    rmse_nb  = np.sqrt(mean_squared_error(y_va, naive_preds))
    mae_nb   = mean_absolute_error(y_va, naive_preds)
    med_nb   = np.median(np.abs(y_va - naive_preds))
    r2_nb    = r2_score(y_va, naive_preds)
    print(f'\n  Naive baseline  (median rpm={naive_rpm:.4f} Ã-- distance):')
    print(f'    RMSE ${rmse_nb:,.2f}  |  MAE ${mae_nb:,.2f}  |  Median AE ${med_nb:,.2f}  |  RÂ² {r2_nb:.4f}')

    # Ensemble
    params = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    print('\n  Training OOT ensemble...')
    m1 = lgb.LGBMRegressor(objective='regression_l1', **params).fit(X_tr, y_tr_rpm)
    m2 = lgb.LGBMRegressor(objective='regression',    **params).fit(X_tr, y_tr_rpm)
    m3 = lgb.LGBMRegressor(objective='regression_l1', **params).fit(X_tr, y_tr)

    p_ens = 0.70 * m1.predict(X_va) * dist_va + 0.15 * m2.predict(X_va) * dist_va + 0.15 * m3.predict(X_va)

    rmse = np.sqrt(mean_squared_error(y_va, p_ens))
    mae  = mean_absolute_error(y_va, p_ens)
    med  = np.median(np.abs(y_va - p_ens))
    r2   = r2_score(y_va, p_ens)
    mape = np.mean(np.abs((y_va - p_ens) / y_va)) * 100.0

    print(f'\n  Tri-Ensemble (70% L1-RPM + 15% L2-RPM + 15% L1-Direct):')
    print(f'    RMSE ${rmse:,.2f}  |  MAE ${mae:,.2f}  |  Median AE ${med:,.2f}  |  RÂ² {r2:.4f}  |  MAPE {mape:.2f}%')
    print(f'\n  Lift over naive baseline:')
    print(f'    RMSE Î” ${rmse_nb - rmse:,.2f}  |  MAE Î” ${mae_nb - mae:,.2f}  |  Median AE Î” ${med_nb - med:,.2f}')

    # â”€â”€ Step 2: Full retrain on all development data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n' + '-' * 60)
    print('STEP 2: FULL RETRAIN ON ALL DEVELOPMENT DATA  (48,000 loads)')
    print('-' * 60)

    X_all    = df_train_feat[FEATURE_COLS]
    y_all    = df_train_feat['posted_rate']
    y_all_rpm = y_all / df_train_feat['distance']

    params_full = dict(n_estimators=1200, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    print('  Fitting Final Model 1: LightGBM L1 (RPM)...')
    fm1 = lgb.LGBMRegressor(objective='regression_l1', **params_full).fit(X_all, y_all_rpm)
    print('  Fitting Final Model 2: LightGBM L2 (RPM)...')
    fm2 = lgb.LGBMRegressor(objective='regression',    **params_full).fit(X_all, y_all_rpm)
    print('  Fitting Final Model 3: LightGBM L1 (Direct)...')
    fm3 = lgb.LGBMRegressor(objective='regression_l1', **params_full).fit(X_all, y_all)

    # â”€â”€ Step 3: Validation predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n' + '-' * 60)
    print('STEP 3: PREDICTING VALIDATION SET  (12,000 loads)')
    print('-' * 60)

    X_v  = df_val_feat[FEATURE_COLS]
    dv   = df_val_feat['distance']
    pv   = np.clip(0.70 * fm1.predict(X_v) * dv + 0.15 * fm2.predict(X_v) * dv + 0.15 * fm3.predict(X_v), 50.0, None)

    sub = pd.DataFrame({'load_id': df_val['load_id'], 'predicted_rate': np.round(pv, 2)})
    sub.to_csv(val_out_path, index=False)
    print(f'  Saved {len(sub):,} predictions â†’ {val_out_path.resolve()}')
    print(sub.head(5).to_string(index=False))

    # â”€â”€ Step 4: December chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n' + '-' * 60)
    print('STEP 4: PREDICTING FIXED DECEMBER SCENARIO  (31 days)')
    print('-' * 60)

    X_d  = df_dec_feat[FEATURE_COLS]
    dd   = df_dec_feat['distance']
    pd_  = np.clip(0.70 * fm1.predict(X_d) * dd + 0.15 * fm2.predict(X_d) * dd + 0.15 * fm3.predict(X_d), 50.0, None)

    # Write to output path; original input CSV is untouched
    df_dec_out = df_dec.copy()
    df_dec_out['predicted_rate'] = np.round(pd_, 2)
    dec_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dec_out.to_csv(dec_out_path, index=False)
    print(f'  Saved December predictions â†’ {dec_out_path.resolve()}')
    print(f'  Rate range: ${pd_.min():.2f} â€“ ${pd_.max():.2f}   Mean: ${pd_.mean():.2f}')
    print('\n  NOTE: The absolute level of December rates is not reliably estimated')
    print('  from Janâ€“Oct training data.  The chart captures weekly day-of-week')
    print('  patterns (Mon ~$811, Wed ~$825) observed across training months,')
    print('  but the true December market level is unknowable without Dec data.')

    print('\n' + sep)
    print('Pipeline completed.  Run scorer:')
    print('  python score.py --predictions validation_predictions.csv \\')
    print('                  --december-predictions data/december_chart_inputs.csv')
    print(sep)


if __name__ == '__main__':
    main()

