import pandas as pd
import numpy as np

# Financial data preprocessing pipeline:
# - Load CSV and standardize column names
# - Validate required columns
# - Handle missing values (closing_price via ffill/bfill; volume via median)
# - Normalize volume with log-scaling
# - Detect price outliers using IQR
# - Engineer returns and lag features
# - Save preprocessed dataset

# --- Load data ---
path = r"C:\Users\savin\OneDrive\Desktop\AI Ass\LAB 17\financial_data.csv"
df = pd.read_csv(path)

print(df.head())
print(df.shape, df.columns.tolist())

# --- Standardize column names ---
# Map common variants to canonical names used downstream
rename_map = {
    'Close': 'closing_price',
    'close': 'closing_price',
    'Volume': 'volume',
    'Date': 'date'
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# Ensure required columns exist
# Early validation prevents cryptic errors later
required_cols = ['closing_price', 'volume']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# --- Handle missing values ---
# closing_price: propagate last/next known value (time-series friendly)
# volume: median is robust to spikes/outliers
print({
    'missing_before': df[required_cols].isna().sum().to_dict()
})

# Convert to numeric safely (non-numeric -> NaN)
df['closing_price'] = pd.to_numeric(df['closing_price'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Impute missing values as described above
df['closing_price'] = df['closing_price'].fillna(method='ffill').fillna(method='bfill')
df['volume'] = df['volume'].fillna(df['volume'].median())

print({
    'missing_after': df[required_cols].isna().sum().to_dict()
})

# --- Normalize volume ---
# log1p handles zeros and stabilizes variance for skewed volumes
df['volume_log'] = np.log1p(df['volume'])
print(df[['volume', 'volume_log']].head())

# --- Outlier detection (IQR) ---
# Classic Tukey rule with 1.5*IQR fences
Q1 = df['closing_price'].quantile(0.25)
Q3 = df['closing_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (df['closing_price'] < lower_bound) | (df['closing_price'] > upper_bound)
df['is_outlier'] = outlier_mask

print({
    'Q1': float(Q1),
    'Q3': float(Q3),
    'IQR': float(IQR),
    'lower_bound': float(lower_bound),
    'upper_bound': float(upper_bound),
    'outliers_count': int(outlier_mask.sum())
})

# Preview outliers
preview_cols = ['closing_price']
if 'date' in df.columns:
    preview_cols.insert(0, 'date')
print(df.loc[outlier_mask, preview_cols])

# --- Feature engineering ---
# Ensure chronological order, then compute returns and lags
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

df['daily_return'] = df['closing_price'].pct_change()

# Lag features
df['closing_price_lag_1'] = df['closing_price'].shift(1)
df['volume_lag_1'] = df['volume'].shift(1)
df['return_lag_1'] = df['daily_return'].shift(1)

df['closing_price_lag_7'] = df['closing_price'].shift(7)
df['volume_lag_7'] = df['volume'].shift(7)
df['return_lag_7'] = df['daily_return'].shift(7)

# 7-day returns
df['return_7day'] = df['closing_price'].pct_change(periods=7)

print(df[['date','closing_price','daily_return','return_lag_1',
          'return_7day','closing_price_lag_1','closing_price_lag_7']].head(10))

# --- Save cleaned/preprocessed data to CSV ---
output_file = 'financial_data_preprocessed.csv'
df.to_csv(output_file, index=False)
print({'csv_created': output_file, 'rows': int(len(df)), 'cols': int(len(df.columns))})
