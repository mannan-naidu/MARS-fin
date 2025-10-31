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

# ### <<< MODIFICATION >>> ###
print("Starting LSTM Single-Output (Price CHANGE/DELTA) Model Script...")

# --- Configuration ---
FILENAME = 'ASIANPAINTS_master_data_engineered.csv'
TARGET_COLUMN = 'Close'
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
LSTM_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 32
# --- Output filenames ---
# ### <<< MODIFICATION >>> ###
RESULTS_FILE = 'lstm_single_output_shap_engineered_delta_results.json'
MODEL_SUMMARY_FILE = 'lstm_single_output_shap_engineered_delta_model_summary.txt'
PREDICTION_PLOT_FILE = 'lstm_single_output_shap_engineered_delta_predictions.png'


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

# --- 2. Feature Selection (SHAP + Engineered Factors) ---
print("Defining SHAP-selected + Engineered features...")
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'sentiment_weighted_mean', 'news_count',
]

BASE_SHAP_FEATURES = [
    'Crude_Oil_Brent', 
    'USD_INR_Exchange_Rate',
    '10Y_Bond_Yield',
    'Policy_Repo_Rate',
    'Net_Profit_Loss_For_the_Period'
]
available_cols = df.columns.tolist()

for base_name in BASE_SHAP_FEATURES:
    val_col = base_name
    chg_col = f"{base_name}_abs_change"
    stale_col = f"{base_name}_days_since_change"

    if (val_col in available_cols and
        chg_col in available_cols and
        stale_col in available_cols):
        
        FEATURES.extend([val_col, chg_col, stale_col])
    else:
        print(f"--- WARNING --- Skipping '{base_name}' as its 3 components are not fully available.")

FEATURES = sorted(list(set(FEATURES)))
print(f"Using {len(FEATURES)} features: {FEATURES}") # 22 features

# --- 3. Preprocessing & Target Creation ---
# ### <<< MODIFICATION >>> ###
print("Preprocessing data and creating TARGET as Price CHANGE (Delta)...")

# Create the new target (y)
TARGET_COLUMN_NEW = 'Close_Change'
df_target_reg = df[[TARGET_COLUMN]].diff().copy()
df_target_reg.columns = [TARGET_COLUMN_NEW]
# This creates one NaN at the beginning

df_features = df[FEATURES].copy()

print("Coercing features to numeric...")
for col in df_features.columns:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

# Concatenate features and the new delta target
all_data = pd.concat([df_features, df_target_reg], axis=1)

# This .dropna() is now CRITICAL: it drops the first row where Close_Change is NaN
all_data = all_data.dropna()

if all_data.empty:
    print("--- ERROR --- DataFrame empty after feature selection/cleaning/target creation.")
    exit()

df_features_clean = all_data[FEATURES]
# Select the new target column
df_target_reg_clean = all_data[[TARGET_COLUMN_NEW]] 

print(f"Data cleaned. Shape: {df_features_clean.shape}")
print(f"Target Reg Shape: {df_target_reg_clean.shape}")

# --- 4. Scaling ---
# ### <<< MODIFICATION >>> ###
print("Scaling data (MinMaxScaler for X, StandardScaler for y_delta)...")
# Use MinMaxScaler for features (X)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_features_clean)

# Use StandardScaler for the target (y), as it's centered around 0
scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(df_target_reg_clean)


# --- 5. Create Sequences ---
print(f"Creating sequences (length {SEQUENCE_LENGTH})...")
X_seq, y_reg_seq = [], []
for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
    y_reg_seq.append(y_reg_scaled[i])

X_seq, y_reg_seq = np.array(X_seq), np.array(y_reg_seq)

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
print("Building Single-Output LSTM model...")
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]), name='input')
lstm_layer = LSTM(LSTM_UNITS, return_sequences=True)(input_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(LSTM_UNITS, return_sequences=False)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
price_branch = Dense(25, activation='relu')(lstm_layer)
price_output = Dense(1, activation='linear', name='price')(price_branch)

model = Model(inputs=input_layer, outputs=price_output)

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
model.summary()

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
    y_reg_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("Training finished.")

# --- 9. Evaluate Model ---
print("Evaluating model...")
losses = model.evaluate(X_test, y_reg_test, verbose=0)
print(f"Test Loss (MSE): {losses[0]:.4f}")
print(f"Test Metric (MAE): {losses[1]:.4f}")


# --- 10. Make Predictions & Inverse Scale ---
print("Making predictions...")
predictions_scaled = model.predict(X_test)

predictions = scaler_y_reg.inverse_transform(predictions_scaled)
actual = scaler_y_reg.inverse_transform(y_reg_test)


# --- 11. Calculate Regression Metrics ---
# ### <<< MODIFICATION >>> ###
print("\n--- Naive Baseline (Predicting 0 Change) ---")
# This is the new baseline to beat.
# It is the error from just guessing "no change" every day.
naive_mae = mean_absolute_error(actual, np.zeros_like(actual))
naive_rmse = np.sqrt(mean_squared_error(actual, np.zeros_like(actual)))
print(f"Naive MAE: {naive_mae:.4f}")
print(f"Naive RMSE: {naive_rmse:.4f} (This is the target to beat)")


print("\n--- Regression Metrics (Price CHANGE Prediction) ---")
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"Test MAE: {mae:.4f} (Price Change)")
print(f"Test RMSE: {rmse:.4f} (Price Change)")

if rmse < naive_rmse:
    print("\n*** SUCCESS: Model RMSE is BETTER than the Naive Baseline. ***")
else:
    print("\n--- INFO: Model RMSE is WORSE than the Naive Baseline. ---")


# --- 12. Save Metrics and Features to File ---
print(f"\nSaving results to {RESULTS_FILE}...")
results = {
    'model_name': f"{FILENAME.split('_')[0]}_LSTM_SingleOutput_SHAP_Delta",
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
        'naive_baseline_rmse': naive_rmse
    }
}

try:
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
except Exception as e:
    print(f"--- ERROR --- Could not save results to JSON: {e}")


# --- 13. Plot Results ---
# ### <<< MODIFICATION >>> ###
print("Plotting results...")
plt.figure(figsize=(14, 7))
plot_index = df_features_clean.index[-len(actual):]
plt.plot(plot_index, actual, label='Actual Price Change', color='blue', alpha=0.7)
plt.plot(plot_index, predictions, label='Predicted Price Change', color='red', linestyle='--', alpha=0.8)
plt.title(f'{FILENAME.split("_")[0]} Price CHANGE Prediction (LSTM SHAP+Engineered Features)')
plt.xlabel('Date')
plt.ylabel('Price Change (Delta)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE)
print(f"Prediction plot saved as {PREDICTION_PLOT_FILE}")
plt.close()

print("\n--- LSTM Delta Model Script Finished ---")