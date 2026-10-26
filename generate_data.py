import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Generate a time series for one year (2023) at an hourly resolution
timestamps = pd.to_datetime(pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="H"))

# Total number of data points
n_points = len(timestamps)

# Create a DataFrame
data = {"timestamp": timestamps}
df = pd.DataFrame(data)

# --- Feature Engineering ---

# 1. Calendar Features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0, Sunday=6
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# 2. Weather Features (Synthetic)
# Temperature (°C): Seasonal + Daily Pattern + Noise
seasonal_temp = 15 - 10 * np.cos(2 * np.pi * (df["day_of_year"] - 30) / 365)
daily_temp_pattern = -5 * np.cos(2 * np.pi * df["hour"] / 24)
noise_temp = np.random.normal(0, 2, n_points)
df["temperature"] = seasonal_temp + daily_temp_pattern + noise_temp

# Relative Humidity (%): Inversely related to temperature + Noise
df["humidity"] = np.clip(70 - (df["temperature"] - 10) * 2 + np.random.normal(0, 5, n_points), 30, 100)

# 3. Energy Consumption (Target Variable)
# Base consumption
base_energy = 150  # kWh

# Influence of time of day (commercial building: peak hours 9-17)
hourly_effect = 50 * np.sin(np.pi * (df["hour"] - 8) / 12)
hourly_effect[((df["hour"] < 8) | (df["hour"] > 20))] = 0

# Weekend effect (lower consumption)
weekend_effect = -80 * df["is_weekend"]

# Temperature effect (heating below 15°C, cooling above 22°C)
heating_effect = 2 * np.maximum(0, 15 - df["temperature"])
cooling_effect = 3 * np.maximum(0, df["temperature"] - 22)

# Random noise
noise_energy = np.random.normal(0, 10, n_points)

# Combine all effects
df["energy_kwh"] = base_energy + hourly_effect + weekend_effect + heating_effect + cooling_effect + noise_energy

# Ensure energy is non-negative
df["energy_kwh"] = np.maximum(0, df["energy_kwh"])

# --- Introduce Missing Data ---
# Randomly introduce 1% missing values in energy and weather columns
for col in ["energy_kwh", "temperature", "humidity"]:
    missing_indices = df.sample(frac=0.01, random_state=42).index
    df.loc[missing_indices, col] = np.nan

# --- Save to CSV ---
# Select and reorder columns for the final dataset
final_df = df[["timestamp", "energy_kwh", "temperature", "humidity"]]
final_df.to_csv("building_data.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

print("Synthetic dataset 'building_data.csv' created successfully.")