import pandas as pd
import os
# IoT sensor cleaning pipeline:
# 1) Load CSV from path
# 2) Coerce numeric readings (temperature/humidity)
# 3) Forward-fill missing values
# 4) Smooth with centered rolling mean to reduce drift
# 5) Standardize (z-score) temperature/humidity
# 6) Encode sensor_id (if present)
# 7) Save updated CSV next to input
# Define file paths (read from input path; write to same folder)
input_file = r"C:\Users\savin\OneDrive\Desktop\AI Ass\LAB 17\iot_sensor.csv"
output_file = os.path.join(os.path.dirname(input_file), 'iot_updated.csv')

# Load the dataset
df = pd.read_csv(input_file)

# Coerce numeric types where applicable (non-numeric -> NaN)
for col in ['temperature', 'humidity']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 1) Handle missing values using forward fill (propagate last known reading)
df = df.fillna(method='ffill')

# 2) Remove sensor drift (centered rolling mean, window=5)
if 'temperature' in df.columns:
    df['temperature'] = df['temperature'].rolling(window=5, min_periods=1, center=True).mean()
if 'humidity' in df.columns:
    df['humidity'] = df['humidity'].rolling(window=5, min_periods=1, center=True).mean()

# 3) Normalize readings using standard scaling (z-score)
for col in ['temperature', 'humidity']:
    if col in df.columns:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std and std != 0:
            df[col + '_scaled'] = (df[col] - mean) / std
        else:
            df[col + '_scaled'] = 0

# 4) Encode categorical sensor IDs
if 'sensor_id' in df.columns:
    df['sensor_id_encoded'] = df['sensor_id'].astype('category').cat.codes

# Save the updated DataFrame to CSV next to input
df.to_csv(output_file, index=False)
print({
    'csv_written': output_file,
    'rows': int(len(df)),
    'cols': int(len(df.columns))
})



