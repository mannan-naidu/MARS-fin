import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# --- Use MinMaxScaler for X, StandardScaler for y ---
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import traceback
import json
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error
)

print("Starting LSTM Single-Output Baseline Model Script (MinMaxScaler)...")

# --- Configuration ---
FILENAME = 'ASIANPAINTS_master_data_engineered.csv'
TARGET_COLUMN = 'Close'
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
LSTM_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 32
# --- Output filenames ---
RESULTS_FILE = 'lstm_single_output_baseline_results.json'
MODEL_SUMMARY_FILE = 'lstm_single_output_baseline_model_summary.txt'
PREDICTION_PLOT_FILE = 'lstm_single_output_baseline_predictions.png'


# --- 1. Load Data ---
print(f"Loading data from {FILENAME}...")
if not os.path.exists(FILENAME):
    print(f"--- ERROR --- File not found: {FILENAME}")
    exit()

try:
    df = pd.read_csv(FILENAME)
    df['date'] = pd.to_datetime(df.pop('date'), dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.set_index('date')
    df = df.sort_index()
    print(f"Data loaded. Full shape: {df.shape}")

except Exception as e:
    print(f"--- ERROR Loading Data ---")
    print(e)
    traceback.print_exc()
    exit()

# --- 2. Feature Selection (Phase 1 Baseline) ---
print("Defining Phase 1 Baseline features...")
PRICE_VOL_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

BASE_FUNDAMENTAL_FEATURES = [
    'Policy_Repo_Rate', '10Y_Bond_Yield', 'Call_Money_Rate', 'CPI_Index',
    'IIP_Index', 'GDP_Growth_QoQ', 'USD_INR_Exchange_Rate',
    'Trade_Exports_USD', 'Trade_Imports_USD', 'FX_Reserves',
    'Crude_Oil_Brent', 'Crude_Oil_WTI',
    'Net_Sales_Income_from_operations', 'Basic_EPS',
    'Net_Profit_Loss_For_the_Period',
    'P_L_Before_Other_Inc_Int_Excpt_Items__Tax', 'Interest',
    'Increase_Decrease_in_Stocks'
]

BASE_SENTIMENT_FEATURES = [
    'sentiment_mean', 'sentiment_weighted_mean', 'news_count',
    'sentiment_mean_roll_avg_7d', 'news_count_roll_sum_7d'
]
BASE_FUNDAMENTAL_FEATURES.extend(BASE_SENTIMENT_FEATURES)

FEATURES = PRICE_VOL_FEATURES.copy()
available_cols = df.columns.tolist()

for base_name in BASE_FUNDAMENTAL_FEATURES:
    val_col = base_name
    chg_col = f"{base_name}_abs_change"
    stale_col = f"{base_name}_days_since_change"

    if (val_col in available_cols and
        chg_col in available_cols and
        stale_col in available_cols):
        
        FEATURES.extend([val_col, chg_col, stale_col])
    elif (val_col in available_cols and
          chg_col not in available_cols and
          stale_col not in available_cols):
        
        if base_name in BASE_SENTIMENT_FEATURES:
            FEATURES.append(val_col)
        else:
            print(f"--- INFO --- '{base_name}' exists but missing engineered features. Skipping.")
            
    elif base_name in PRICE_VOL_FEATURES:
        pass
    else:
        print(f"--- WARNING --- Skipping '{base_name}' (or its components) as it's not fully available.")

FEATURES = sorted(list(set(FEATURES)))
print(f"Using {len(FEATURES)} features for Phase 1 Baseline: {FEATURES}")

# --- 3. Preprocessing & Target Creation ---
print("Preprocessing data and creating targets...")

# Target 1: Regression (Predict 'Close' price)
df_target_reg = df[[TARGET_COLUMN]].copy()

# Select features for X
df_features = df[FEATURES].copy()

# Clean the features *before* concatenating
print("Coercing features to numeric...")
for col in df_features.columns:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

# Now concatenate the already-clean features with the target
all_data = pd.concat([df_features, df_target_reg], axis=1)
all_data = all_data.dropna()

if all_data.empty:
    print("--- ERROR --- DataFrame empty after feature selection/cleaning/target creation.")
    exit()

# Separate X and y dataframes *after* cleaning
df_features_clean = all_data[FEATURES]

# Explicitly select only the FIRST column named 'Close'
# This ensures df_target_reg_clean is (N, 1), not (N, 2)
df_target_reg_clean = all_data[[TARGET_COLUMN]].iloc[:, 0:1] 

print(f"Data cleaned. Shape: {df_features_clean.shape}")
print(f"Target Reg Shape: {df_target_reg_clean.shape}")

# --- 4. Scaling ---
# ### <<< MODIFICATION >>> Using MinMaxScaler for X, StandardScaler for y
print("Scaling data (MinMaxScaler for X, StandardScaler for y)...")
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_features_clean)

scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(df_target_reg_clean)


# --- 5. Create Sequences ---
print(f"Creating sequences (length {SEQUENCE_LENGTH})...")
X_seq, y_reg_seq = [], []

for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
    y_reg_seq.append(y_reg_scaled[i])

X_seq = np.array(X_seq)
y_reg_seq = np.array(y_reg_seq)

if X_seq.shape[0] == 0:
    print(f"--- ERROR --- Not enough data ({len(X_scaled)} rows) for sequences.")
    exit()
print(f"Sequences created. X shape: {X_seq.shape}")
print(f"Regression y shape: {y_reg_seq.shape}")


# --- 6. Split Data ---
print(f"Splitting data (Test size: {TEST_SIZE})...")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_seq, y_reg_seq, 
    test_size=TEST_SIZE, 
    shuffle=False
)

print(f"Train shapes: X={X_train.shape}, y_reg={y_reg_train.shape}")
print(f"Test shapes: X={X_test.shape}, y_reg={y_reg_test.shape}")


# --- 7. Build LSTM Model (Single-Output) ---
# ### <<< MODIFICATION >>> Single-Output Regression Model
print("Building Single-Output LSTM model...")

# Input Layer
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]), name='input')

# LSTM Layers
lstm_layer = LSTM(LSTM_UNITS, return_sequences=True)(input_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(LSTM_UNITS, return_sequences=False)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

# Output Head: Regression (Price)
price_branch = Dense(25, activation='relu')(lstm_layer)
price_output = Dense(1, activation='linear', name='price')(price_branch)

# Build the Model
model = Model(inputs=input_layer, outputs=price_output)

# Compile with single loss
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
model.summary()
# --- End Model Build ---

# --- Save model summary to file ---
try:
    with open(MODEL_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {MODEL_SUMMARY_FILE}")
except Exception as e:
    print(f"--- WARNING --- Could not save model summary: {e}")

# --- 8. Train Model ---
print("Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, 
    y_reg_train,  # ### <<< MODIFICATION >>> Pass the regression target directly
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("Training finished.")

# --- 9. Evaluate Model ---
print("Evaluating model...")
# ### <<< MODIFICATION >>> model.evaluate now returns [loss, mae]
losses = model.evaluate(X_test, y_reg_test, verbose=0)
print(f"Test Loss (MSE): {losses[0]:.4f}")
print(f"Test Metric (MAE): {losses[1]:.4f}")


# --- 10. Make Predictions & Inverse Scale ---
print("Making predictions...")
# ### <<< MODIFICATION >>> model.predict returns a single array
predictions_scaled = model.predict(X_test)

# --- Inverse transform for REGRESSION results ---
predictions = scaler_y_reg.inverse_transform(predictions_scaled)
actual = scaler_y_reg.inverse_transform(y_reg_test)


# --- 11. Calculate Regression Metrics ---
print("\n--- Regression Metrics (Price Prediction) ---")
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"Test MAE: {mae:.4f} (Price)")
print(f"Test RMSE: {rmse:.4f} (Price)")

# --- 12. Save Metrics and Features to File ---
print(f"\nSaving results to {RESULTS_FILE}...")
results = {
    'model_name': f"{FILENAME.split('_')[0]}_LSTM_SingleOutput_Baseline",
    'features_used': FEATURES,
    'feature_count': len(FEATURES),
    'evaluation_losses_scaled': {
        'test_loss_mse': losses[0],
        'test_metric_mae': losses[1]
    },
    'regression_metrics_unscaled': {
        'mae': mae,
        'rmse': rmse
    }
}

try:
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
except Exception as e:
    print(f"--- ERROR --- Could not save results to JSON: {e}")


# --- 13. Plot Results ---
print("Plotting results...")

# Plot 1: Predictions vs Actual (Regression)
plt.figure(figsize=(14, 7))
plot_index = df_features_clean.index[-len(actual):]
plt.plot(plot_index, actual, label='Actual Close Price', color='blue', alpha=0.7)
plt.plot(plot_index, predictions, label='Predicted Close Price', color='red', linestyle='--')
plt.title(f'{FILENAME.split("_")[0]} Close Price Prediction (LSTM Single-Output Baseline)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE)
print(f"Prediction plot saved as {PREDICTION_PLOT_FILE}")
plt.close()

print("\n--- LSTM Single-Output Script Finished ---")