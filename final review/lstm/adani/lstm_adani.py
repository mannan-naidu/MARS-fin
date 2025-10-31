import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import traceback

print("Starting LSTM Model Script...")

# --- Configuration ---
FILENAME = 'ADANIENT_master_data.csv' # Or 'ADANIENT_master_data.csv'
TARGET_COLUMN = 'Close'
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
LSTM_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 32

# --- Feature Selection ---
# --- THIS IS THE FIX ---
# Removed the '_quarterly' suffix from the quarterly columns
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'sentiment_weighted_mean', 'news_count', 'sentiment_std',
    'sentiment_weighted_mean_roll_avg_7d', 'news_count_roll_sum_7d',
    'Policy_Repo_Rate', '10Y_Bond_Yield', 'USD_INR_Exchange_Rate', 'Crude_Oil_Brent',
    # Corrected quarterly column names
    'Net_Sales_Income_from_operations',
    'Net_Profit_Loss_For_the_Period',
    'Basic_EPS'
]
# --- END OF FIX ---

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
    for col in FEATURES:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    df_features = df_features.dropna()

    if df_features.empty:
        print("--- ERROR --- DataFrame empty after feature selection/cleaning.")
        exit()

    print(f"Data loaded successfully. Shape: {df_features.shape}")
    print(f"Using features: {FEATURES}")

except Exception as e:
    print(f"--- ERROR Loading Data ---")
    print(e)
    traceback.print_exc()
    exit()

# --- 2. Preprocessing ---
print("Preprocessing data (scaling)...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_features)

target_col_index = df_features.columns.tolist().index(TARGET_COLUMN)
print(f"Target column '{TARGET_COLUMN}' index: {target_col_index}")

# --- Create Sequences ---
print(f"Creating sequences (length {SEQUENCE_LENGTH})...")
X, y = [], []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(scaled_data[i, target_col_index])
X, y = np.array(X), np.array(y)
if X.shape[0] == 0:
    print(f"--- ERROR --- Not enough data ({len(scaled_data)} rows) for sequences.")
    exit()
print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# --- 3. Split Data ---
print(f"Splitting data (Test size: {TEST_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

# --- 4. Build LSTM Model ---
print("Building LSTM model...")
model = Sequential([
    LSTM(LSTM_UNITS, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(LSTM_UNITS, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 5. Train Model ---
print("Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("Training finished.")

# --- 6. Evaluate Model ---
print("Evaluating model...")
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss}")

# --- 7. Make Predictions & Inverse Scale ---
print("Making predictions...")
predictions_scaled = model.predict(X_test)
dummy_predictions = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
dummy_predictions[:, target_col_index] = predictions_scaled.flatten()
predictions = scaler.inverse_transform(dummy_predictions)[:, target_col_index]

dummy_y_test = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_y_test[:, target_col_index] = y_test.flatten()
actual = scaler.inverse_transform(dummy_y_test)[:, target_col_index]

# --- Calculate Metrics ---
mae = np.mean(np.abs(predictions - actual))
rmse = np.sqrt(np.mean((predictions - actual)**2))
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# --- 8. Plot Results ---
print("Plotting results...")
plt.figure(figsize=(14, 7))
plot_index = df_features.index[-len(actual):]
plt.plot(plot_index, actual, label='Actual Close Price', color='blue', alpha=0.7)
plt.plot(plot_index, predictions, label='Predicted Close Price', color='red', linestyle='--')
plt.title(f'{FILENAME.split("_")[0]} Close Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('lstm_predictions_vs_actual.png')
print("Plot saved as lstm_predictions_vs_actual.png")

print("\n--- LSTM Script Finished ---")