import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

def test_december_models():
    df_train = pd.read_csv("train-test.csv")
    df_val = pd.read_csv("validation.csv")
    df_dec = pd.read_csv("december-chart-inputs.csv")
    
    # Process dates
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_val['date'] = pd.to_datetime(df_val['date'])
    df_dec['date'] = pd.to_datetime(df_dec['date'])
    
    # Check daily market index in December from validation
    val_dec = df_val[(df_val['date'] >= '2025-12-01') & (df_val['date'] <= '2025-12-31')]
    daily_mi = val_dec.groupby(val_dec['date'].dt.date)['market_index'].mean()
    print("Daily MI count in Dec val:", len(daily_mi))
    
    # City coordinates mapping
    city_coords = {}
    for _, row in pd.concat([df_train[['pickup', 'pickup_lat', 'pickup_lon']], 
                            df_val[['pickup', 'pickup_lat', 'pickup_lon']]]).drop_duplicates('pickup').iterrows():
        city_coords[row['pickup']] = (row['pickup_lat'], row['pickup_lon'])
        
    print(f"Total mapped cities: {len(city_coords)}")
    
    # Check quote signal for Lexington -> Fort Wayne
    lex_fw = pd.concat([
        df_train[(df_train['pickup']=='Lexington') & (df_train['delivery']=='Fort Wayne')],
        df_val[(df_val['pickup']=='Lexington') & (df_val['delivery']=='Fort Wayne')]
    ])
    mean_qs_lex_fw = lex_fw[lex_fw['equipment'] == 'Dry Van']['quote_signal'].mean()
    print(f"Mean QS for Lexington -> Fort Wayne Dry Van: {mean_qs_lex_fw:.4f}")
    
    # Let's inspect how date features affect the prediction
    def add_features(df):
        d = df.copy()
        d['month'] = d['date'].dt.month
        d['day'] = d['date'].dt.day
        d['dayofweek'] = d['date'].dt.dayofweek
        d['dayofyear'] = d['date'].dt.dayofyear
        d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
        d['week'] = d['date'].dt.isocalendar().week.astype(int)
        
        # Sine/Cosine cyclical date features
        d['sin_dayofweek'] = np.sin(2 * np.pi * d['dayofweek'] / 7)
        d['cos_dayofweek'] = np.cos(2 * np.pi * d['dayofweek'] / 7)
        d['sin_dayofyear'] = np.sin(2 * np.pi * d['dayofyear'] / 365.25)
        d['cos_dayofyear'] = np.cos(2 * np.pi * d['dayofyear'] / 365.25)
        
        return d

    df_train_feat = add_features(df_train)
    df_dec_feat = add_features(df_dec)
    
    # Map lat/lon for dec
    df_dec_feat['pickup_lat'] = city_coords['Lexington'][0]
    df_dec_feat['pickup_lon'] = city_coords['Lexington'][1]
    df_dec_feat['delivery_lat'] = city_coords['Fort Wayne'][0]
    df_dec_feat['delivery_lon'] = city_coords['Fort Wayne'][1]
    
    # Encode categoricals
    df_train_feat['equipment'] = df_train_feat['equipment'].astype('category')
    df_dec_feat['equipment'] = pd.Categorical(df_dec_feat['equipment'], categories=df_train_feat['equipment'].cat.categories)

    features = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'equipment', 'weight',
        'month', 'day', 'dayofweek', 'dayofyear', 'is_weekend',
        'sin_dayofweek', 'cos_dayofweek', 'sin_dayofyear', 'cos_dayofyear'
    ]
    
    # Model without signals
    model_no_signals = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    model_no_signals.fit(df_train_feat[features], df_train_feat['posted_rate'])
    dec_preds_no_signals = model_no_signals.predict(df_dec_feat[features])
    
    print("December predictions (Model without signals):")
    print(pd.DataFrame({'date': df_dec['date'], 'pred': dec_preds_no_signals}).head(10))
    print("Summary of dec preds without signals:")
    print(pd.Series(dec_preds_no_signals).describe())

if __name__ == "__main__":
    test_december_models()
