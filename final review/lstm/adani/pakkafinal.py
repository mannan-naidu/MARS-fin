import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import traceback
import json
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error
)

# ### <<< FINAL MODEL SCRIPT >>> ###
print("Starting Final GRU Single-Output (Delta) Model - PURE TECH, SENTIMENT & NEWS CATEGORIES...")

# --- 1. Configuration & Tuning ---
# ### <<< MODIFICATION: Using our NEW final dataset >>> ###
FILENAME = 'ADANIENT_master_data_FINAL.csv'
TARGET_COLUMN = 'Close'
STOCK_TICKER = FILENAME.split('_')[0]

TUNING_VARS = {
    "SEQUENCE_LENGTH": 30,
    "LSTM_UNITS": 100,
    "DROPOUT": 0.2,
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "TEST_SIZE": 0.2,
    "USE_GRU": True,
}
MODEL_ARCH_NAME = "GRU" if TUNING_VARS["USE_GRU"] else "LSTM"

# --- Output filenames ---
RESULTS_FILE = f'{STOCK_TICKER}_FINAL_model_results.json'
MODEL_SUMMARY_FILE = f'{STOCK_TICKER}_FINAL_model_summary.txt'
PREDICTION_PLOT_FILE = f'{STOCK_TICKER}_FINAL_model_predictions.png'
SHAP_PLOT_FILE = f'{STOCK_TICKER}_FINAL_model_shap_importance.png' # For SHAP script later

# --- 2. Load Data ---
print(f"Loading data from {FILENAME}...")
if not os.path.exists(FILENAME):
    print(f"--- ERROR --- File not found: {FILENAME}")
    exit()

try:
    df = pd.read_csv(FILENAME)
    # ### Corrected date parsing (no dayfirst=True) ###
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date'])
    df = df.set_index('date')
    df = df.sort_index()
    print(f"Data loaded. Full shape: {df.shape}")
except Exception as e:
    print(f"--- ERROR Loading Data ---: {e}")
    exit()

# --- 3. Add Technical Analysis (TA) Features ---
print("Calculating Technical Analysis features...")
TA_FEATURES = []

# RSI
try:
    df['TA_RSI_14'] = ta.rsi(df['Close'], length=14)
    TA_FEATURES.append('TA_RSI_14')
except Exception as e:
    print(f"--- WARNING --- Failed to calculate RSI: {e}")

# MACD
try:
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['TA_MACD_hist'] = macd['MACDh_12_26_9']
    TA_FEATURES.append('TA_MACD_hist')
except Exception as e:
    print(f"--- WARNING --- Failed to calculate MACD: {e}")

# Bollinger Bands (Robust fix)
try:
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['TA_BB_Lower'] = bbands['BBL_20_2.0']
    df['TA_BB_Mid'] = bbands['BBM_20_2.0']
    df['TA_BB_Upper'] = bbands['BBU_20_2.0']
    TA_FEATURES.extend(['TA_BB_Lower', 'TA_BB_Mid', 'TA_BB_Upper'])
except Exception as e:
    print(f"--- WARNING --- Failed to calculate Bollinger Bands: {e}")

print(f"TA features calculated: {TA_FEATURES}")

# --- 4. Feature Selection (Tech + Sentiment + Categories) ---
print("Defining 'Pure Tech, Sentiment & News Categories' features...")
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
]
FEATURES.extend(TA_FEATURES) # Add the TA features
available_cols = df.columns.tolist()

# Add ALL available sentiment features
BASE_SENTIMENT_FEATURES = [
    'sentiment_mean', 'sentiment_weighted_mean', 'news_count',
    'sentiment_mean_roll_avg_3d', 'sentiment_weighted_mean_roll_avg_3d', 'news_count_roll_sum_3d',
    'sentiment_mean_roll_avg_7d', 'sentiment_weighted_mean_roll_avg_7d', 'news_count_roll_sum_7d',
    'sentiment_mean_roll_avg_14d', 'sentiment_weighted_mean_roll_avg_14d', 'news_count_roll_sum_14d'
]
for feature in BASE_SENTIMENT_FEATURES:
    if feature in available_cols:
        FEATURES.append(feature)

# ### <<< MODIFICATION: Add our NEW multi-hot categories >>> ###
MULTI_HOT_CATEGORIES = [
    'is_fraud_allegation',
    'is_regulatory_probe',
    'is_financial_positive',
    'is_financial_negative',
    'is_company_response'
]
for feature in MULTI_HOT_CATEGORIES:
    if feature in available_cols:
        FEATURES.append(feature)
    else:
        print(f"--- WARNING --- Multi-hot feature '{feature}' not found in CSV!")

# We are intentionally *ignoring* all macro/fundamental event features

FEATURES = sorted(list(set(FEATURES)))
print(f"Using {len(FEATURES)} features: {FEATURES}")

# --- 5. Preprocessing & Target Creation ---
print("Preprocessing data and creating REGRESSION target (Price Change)...")
TARGET_REG_COL = 'Close_Change'
df_target_reg = df[[TARGET_COLUMN]].diff().copy()
df_target_reg.columns = [TARGET_REG_COL] 
df_features = df[FEATURES].copy()

print("Coercing features to numeric...")
for col in df_features.columns:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

all_data = pd.concat([df_features, df_target_reg], axis=1)
# .dropna() is critical - removes NaNs from .diff() and TA indicators
all_data = all_data.dropna()

if all_data.empty:
    print("--- ERROR --- DataFrame empty after feature/target creation and NaN drop.")
    exit()

df_features_clean = all_data[FEATURES]
df_target_reg_clean = all_data[[TARGET_REG_COL]]
print(f"Data cleaned. Feature Shape: {df_features_clean.shape}, Target Shape: {df_target_reg_clean.shape}")

# --- 6. Scaling ---
print("Scaling data (MinMaxScaler for X, StandardScaler for y_delta)...")
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_features_clean)
scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(df_target_reg_clean)

# --- 7. Create Sequences ---
print(f"Creating sequences (length {TUNING_VARS['SEQUENCE_LENGTH']})...")
X_seq, y_reg_seq = [], []
for i in range(TUNING_VARS['SEQUENCE_LENGTH'], len(X_scaled)):
    X_seq.append(X_scaled[i-TUNING_VARS['SEQUENCE_LENGTH']:i])
    y_reg_seq.append(y_reg_scaled[i])
X_seq, y_reg_seq = np.array(X_seq), np.array(y_reg_seq)
if X_seq.shape[0] == 0:
    print(f"--- ERROR --- Not enough data for sequences.")
    exit()
print(f"Sequences created. X shape: {X_seq.shape}")

# --- 8. Split Data ---
print(f"Splitting data (Test size: {TUNING_VARS['TEST_SIZE']})...")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_seq, y_reg_seq, 
    test_size=TUNING_VARS['TEST_SIZE'], 
    shuffle=False
)
print(f"Train shapes: X={X_train.shape}, y_reg={y_reg_train.shape}")
print(f"Test shapes: X={X_test.shape}, y_reg={y_reg_test.shape}")

# --- 9. Build Model ---
print(f"Building Single-Output {MODEL_ARCH_NAME} model...")
RecurrentLayer = GRU if TUNING_VARS["USE_GRU"] else LSTM
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]), name='input')
shared_base = RecurrentLayer(TUNING_VARS['LSTM_UNITS'], return_sequences=True)(input_layer)
shared_base = Dropout(TUNING_VARS['DROPOUT'])(shared_base)
shared_base = RecurrentLayer(TUNING_VARS['LSTM_UNITS'], return_sequences=False)(shared_base)
shared_base = Dropout(TUNING_VARS['DROPOUT'])(shared_base)
price_branch = Dense(25, activation='relu')(shared_base)
price_output = Dense(1, activation='linear', name='price')(price_branch)
model = Model(inputs=input_layer, outputs=price_output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

try:
    with open(MODEL_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {MODEL_SUMMARY_FILE}")
except Exception as e:
    print(f"--- WARNING --- Could not save model summary: {e}")

# --- 10. Train Model ---
print("Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, 
    y_reg_train,
    batch_size=TUNING_VARS['BATCH_SIZE'],
    epochs=TUNING_VARS['EPOCHS'],
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("Training finished.")

# --- 11. Evaluate Model ---
print("Evaluating model...")
losses = model.evaluate(X_test, y_reg_test, verbose=0)
print(f"Test Loss (MSE): {losses[0]:.4f}")
print(f"Test Metric (MAE): {losses[1]:.4f}")

# --- 12. Make Predictions & Inverse Scale ---
print("Making predictions...")
predictions_scaled = model.predict(X_test)
predictions = scaler_y_reg.inverse_transform(predictions_scaled)
actual = scaler_y_reg.inverse_transform(y_reg_test)

# --- 13. Calculate REGRESSION Metrics ---
print("\n--- Naive Baseline (Predicting 0 Change) ---")
# Get the correct naive baseline from the *actual* test set data
# This is critical because the test set changes size as NaNs are dropped
actual_test_data_for_baseline = all_data[[TARGET_REG_COL]].iloc[-len(actual):]
naive_mae = mean_absolute_error(actual_test_data_for_baseline, np.zeros_like(actual_test_data_for_baseline))
naive_rmse = np.sqrt(mean_squared_error(actual_test_data_for_baseline, np.zeros_like(actual_test_data_for_baseline)))
print(f"Naive MAE: {naive_mae:.4f}")
print(f"Naive RMSE: {naive_rmse:.4f} (This is the target to beat)")

print("\n--- Regression Metrics (Price CHANGE Prediction) ---")
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"Test MAE: {mae:.4f} (Price Change)")
print(f"Test RMSE: {rmse:.4f} (Price Change)")

if rmse < naive_rmse:
    print("\n*** SUCCESS: Model REGRESSION RMSE is BETTER than the Naive Baseline. ***")
else:
    print("\n--- INFO: Model REGRESSION RMSE is WORSE than the Naive Baseline. ---")

# --- 14. Save Metrics and Features to File ---
print(f"\nSaving results to {RESULTS_FILE}...")
results = {
    'model_name': f"{STOCK_TICKER}_SingleOutput_Delta_PureTechSentimentCategories",
    'tuning_params': TUNING_VARS,
    'model_architecture': MODEL_ARCH_NAME,
    'features_used': FEATURES,
    'feature_count': len(FEATURES),
    'evaluation_losses_scaled': {
        'test_loss_mse': losses[0],
        'test_metric_mae': losses[1]
    },
    'regression_metrics_unscaled_delta': {
        'model_mae': mae,
        'model_rmse': rmse,
        'naive_baseline_mae': naive_mae,
        'naive_baseline_rmse': naive_rmse,
        'improvement_over_naive_rmse_pct': (naive_rmse - rmse) / naive_rmse * 100
    }
}

try:
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
except Exception as e:
    print(f"--- ERROR --- Could not save results to JSON: {e}")

# --- 15. Plot Results ---
print("Plotting results...")
plt.figure(figsize=(14, 7))
plot_index = df_features_clean.index[-len(actual):]
plt.plot(plot_index, actual, label='Actual Price Change', color='blue', alpha=0.7)
plt.plot(plot_index, predictions, label='Predicted Price Change', color='red', linestyle='--', alpha=0.8)
plt.title(f'{STOCK_TICKER} Price CHANGE Prediction ({MODEL_ARCH_NAME} - Tech, Sentiment & News Categories)')
plt.xlabel('Date')
plt.ylabel('Price Change (Delta)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE)
# ### <<< MODIFICATION: Fixed typo >>> ###
print(f"Prediction plot saved as {PREDICTION_PLOT_FILE}")
plt.close()

print(f"\n--- {MODEL_ARCH_NAME} Single-Output Delta Model w/TA Script Finished ---")