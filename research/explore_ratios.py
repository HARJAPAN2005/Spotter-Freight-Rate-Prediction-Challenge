import pandas as pd
import numpy as np

def explore_ratios():
    df = pd.read_csv("train-test.csv")
    df['base'] = df['distance'] * df['quote_signal']
    df['ratio'] = df['posted_rate'] / df['base']
    
    print("Ratio percentiles:")
    print(np.percentile(df['ratio'], [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 100]))
    
    # What fraction of rows have ratio near 1.0?
    near_1 = (df['ratio'] >= 0.95) & (df['ratio'] <= 1.05)
    print(f"Fraction with ratio in [0.95, 1.05]: {near_1.mean():.4f} ({near_1.sum()} rows)")
    
    near_1_loose = (df['ratio'] >= 0.90) & (df['ratio'] <= 1.10)
    print(f"Fraction with ratio in [0.90, 1.10]: {near_1_loose.mean():.4f} ({near_1_loose.sum()} rows)")
    
    # Let's inspect the distribution of ratio!
    # Is there a bimodal or multimodal distribution?
    hist, bin_edges = np.histogram(df['ratio'], bins=30)
    print("\nHistogram of ratio:")
    for i in range(len(hist)):
        print(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {hist[i]}")
        
    # Check if ratio depends on date, equipment, market_index, weight
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    print("\nMean and median ratio by month:")
    print(df.groupby('month')['ratio'].agg(['count', 'mean', 'median', 'std', lambda x: (x > 1.5).mean()]))
    
    print("\nMean and median ratio by equipment:")
    print(df.groupby('equipment')['ratio'].agg(['count', 'mean', 'median', 'std', lambda x: (x > 1.5).mean()]))

if __name__ == "__main__":
    explore_ratios()
