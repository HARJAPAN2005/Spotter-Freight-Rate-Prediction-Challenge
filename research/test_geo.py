import pandas as pd
import numpy as np

def haversine_np(lat1, lon1, lat2, lon2):
    # Earth radius in miles
    R = 3958.8
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def test_geo_features():
    df_train = pd.read_csv("train-test.csv")
    df_val = pd.read_csv("validation.csv")
    
    for df in [df_train, df_val]:
        df['haversine'] = haversine_np(df['pickup_lat'], df['pickup_lon'], df['delivery_lat'], df['delivery_lon'])
        df['tortuosity'] = df['distance'] / (df['haversine'] + 1e-5)
        
    print("Train Haversine correlation with distance:", df_train['distance'].corr(df_train['haversine']))
    print("Train Tortuosity describe:\n", df_train['tortuosity'].describe())
    print("Val Tortuosity describe:\n", df_val['tortuosity'].describe())

if __name__ == "__main__":
    test_geo_features()
