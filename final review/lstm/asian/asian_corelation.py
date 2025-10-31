import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import traceback

print("Starting Feature Correlation Matrix Script...")

# --- Configuration ---
FILENAME = 'ASIANPAINTS_master_data.csv' # Or 'ADANIENT_master_data.csv'
OUTPUT_IMAGE = 'feature_correlation_heatmap.png'

# --- Feature Selection ---
# Use the SAME features as your successful LSTM run
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'sentiment_weighted_mean', 'news_count', 'sentiment_std',
    'sentiment_weighted_mean_roll_avg_7d', 'news_count_roll_sum_7d',
    'Policy_Repo_Rate', '10Y_Bond_Yield', 'USD_INR_Exchange_Rate', 'Crude_Oil_Brent',
    # Corrected quarterly column names (without suffix)
    'Net_Sales_Income_from_operations',
    'Net_Profit_Loss_For_the_Period',
    'Basic_EPS'
]

# --- 1. Load Data ---
print(f"Loading data from {FILENAME}...")
if not os.path.exists(FILENAME):
    print(f"--- ERROR --- File not found: {FILENAME}")
    exit()

try:
    df = pd.read_csv(FILENAME)
    # Use dayfirst=True based on previous errors
    df['date'] = pd.to_datetime(df.pop('date'), dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.set_index('date')
    df = df.sort_index()

    # --- Check if all selected FEATURES exist ---
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"--- ERROR --- Features missing: {missing_features}")
        available_cols = df.columns.tolist()
        print(f"Available columns ({len(available_cols)}): {available_cols}")
        print("\nPlease correct the FEATURES list.")
        exit()

    df_features = df[FEATURES].copy()
    # Convert all feature columns to numeric, coercing errors
    for col in FEATURES:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    df_features = df_features.dropna() # Drop rows with NaNs AFTER numeric conversion

    if df_features.empty:
        print("--- ERROR --- DataFrame empty after feature selection/cleaning.")
        exit()

    print(f"Data loaded successfully. Shape: {df_features.shape}")

except Exception as e:
    print(f"--- ERROR Loading Data ---")
    print(e)
    traceback.print_exc()
    exit()

# --- 2. Calculate Correlation Matrix ---
# Note: Correlation calculation itself works on the raw feature values.
# No special normalization based on 'importance' is needed or standard practice here.
# Scaling (like MinMax or Standard) is for model training, not correlation analysis.
print("Calculating correlation matrix...")
correlation_matrix = df_features.corr()

# Optional: Print the matrix values
# print("\nCorrelation Matrix Values:")
# print(correlation_matrix)

# --- 3. Visualize Correlation Matrix as Heatmap ---
print("Generating heatmap...")
plt.figure(figsize=(16, 12)) # Adjust size as needed for readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
plt.title(f'Feature Correlation Matrix for {FILENAME.split("_")[0]} LSTM Features', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout() # Adjust layout to prevent labels overlapping

# --- 4. Save Heatmap ---
plt.savefig(OUTPUT_IMAGE)
print(f"Heatmap saved as {OUTPUT_IMAGE}")

print("\n--- Script Finished ---")