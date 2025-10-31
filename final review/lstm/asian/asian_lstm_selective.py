import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# --- Use StandardScaler ---
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import traceback
import json  # Added for saving metrics
import seaborn as sns  # Added for confusion matrix plotting
from sklearn.metrics import (  # Added for metrics
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)


print("Starting LSTM Model Script (using SHAP features & StandardScaler)...")

# --- Configuration ---
FILENAME = 'ASIANPAINTS_master_data.csv' # Or 'ADANIENT_master_data.csv'
TARGET_COLUMN = 'Close'
SEQUENCE_LENGTH = 30 # You can tune this
TEST_SIZE = 0.2
LSTM_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 32
# --- Output filenames ---
RESULTS_FILE = 'lstm_shap_features_results.json'
MODEL_SUMMARY_FILE = 'lstm_shap_model_summary.txt'
PREDICTION_PLOT_FILE = 'lstm_shap_features_predictions.png'
CM_PLOT_FILE = 'lstm_shap_features_confusion_matrix.png'

# --- Feature Selection (Based on SHAP plot) ---
# Select top features identified
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', # Top price/volume
    'Crude_Oil_Brent',                       # Top Macro
    'USD_INR_Exchange_Rate',                 # Next Macro
    '10Y_Bond_Yield',                        # Next Macro
    'sentiment_weighted_mean',               # Top Sentiment
    'news_count',                            # Next Sentiment (often correlated with impact)
    'Policy_Repo_Rate',                      # Next Macro
    # Include one key quarterly metric if desired, though lower importance
    'Net_Profit_Loss_For_the_Period'
]
# We removed less important sentiment (std, rolling), macro (CPI, IIP etc.), and quarterly

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
    print(f"Using top {len(FEATURES)} features based on SHAP: {FEATURES}")

except Exception as e:
    print(f"--- ERROR Loading Data ---")
    print(e)
    traceback.print_exc()
    exit()

# --- 2. Preprocessing ---
print("Preprocessing data (StandardScaler)...")
# --- Use StandardScaler ---
scaler = StandardScaler()
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

# --- Save model summary to file ---
try:
    with open(MODEL_SUMMARY_FILE, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {MODEL_SUMMARY_FILE}")
except Exception as e:
    print(f"--- WARNING --- Could not save model summary: {e}")

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

# --- Inverse transform using StandardScaler ---
# Create dummy array matching the scaled_data shape
dummy_predictions = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
dummy_predictions[:, target_col_index] = predictions_scaled.flatten()
predictions = scaler.inverse_transform(dummy_predictions)[:, target_col_index]

dummy_y_test = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_y_test[:, target_col_index] = y_test.flatten()
actual = scaler.inverse_transform(dummy_y_test)[:, target_col_index]

# Get the last actual close from the training set for directional comparison
dummy_y_train = np.zeros((len(y_train), scaled_data.shape[1]))
dummy_y_train[:, target_col_index] = y_train.flatten()
actual_train = scaler.inverse_transform(dummy_y_train)[:, target_col_index]
last_train_actual_close = actual_train[-1]
# --- End inverse transform ---


# --- 8. Calculate Regression Metrics ---
print("\n--- Regression Metrics ---")
mae = np.mean(np.abs(predictions - actual))
rmse = np.sqrt(np.mean((predictions - actual)**2))
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# --- 9. Calculate Classification Metrics (Directional Accuracy) ---
# This is a regression problem, so a standard confusion matrix
# isn't applicable. We derive one by checking if the model
# correctly predicted the *direction* of the price move (Up vs. Down/Same)
# compared to the *previous day's* actual close.

print("\n--- Classification Metrics (Directional) ---")
# Create array of previous day's actual close prices
previous_actual_closes = np.roll(actual, 1)
previous_actual_closes[0] = last_train_actual_close # Set first element to last train close

# Determine actual and predicted direction (1 for Up, 0 for Down/Same)
actual_direction = (actual > previous_actual_closes).astype(int)
predicted_direction = (predictions > previous_actual_closes).astype(int)

# Calculate metrics
accuracy = accuracy_score(actual_direction, predicted_direction)
precision = precision_score(actual_direction, predicted_direction, zero_division=0)
recall = recall_score(actual_direction, predicted_direction, zero_division=0)
f1 = f1_score(actual_direction, predicted_direction, zero_division=0)
cm = confusion_matrix(actual_direction, predicted_direction)
report = classification_report(
    actual_direction,
    predicted_direction,
    target_names=['Down/Same (0)', 'Up (1)'],
    zero_division=0
)

print(f"Directional Accuracy: {accuracy:.4f}")
print(f"Directional Precision (for 'Up'): {precision:.4f}")
print(f"Directional Recall (for 'Up'): {recall:.4f}")
print(f"Directional F1-Score (for 'Up'): {f1:.4f}")
print("Confusion Matrix (Directional):")
print(cm)
print("\nClassification Report (Directional):")
print(report)

# --- 10. Save Metrics, Features, and CM to File ---
print(f"\nSaving results to {RESULTS_FILE}...")
results = {
    'model_name': f"{FILENAME.split('_')[0]}_LSTM_SHAP_Features",
    'features_used': FEATURES,
    'regression_metrics': {
        'test_loss_mse': loss,
        'mae': mae,
        'rmse': rmse
    },
    'classification_metrics_directional': {
        'note': "Derived by predicting direction (Up vs. Down/Same) vs. previous day's actual close.",
        'accuracy': accuracy,
        'precision_up': precision,
        'recall_up': recall,
        'f1_score_up': f1,
        'confusion_matrix': cm.tolist(), # Convert numpy array to list for JSON
        'classification_report': report
    }
}

try:
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
except Exception as e:
    print(f"--- ERROR --- Could not save results to JSON: {e}")


# --- 11. Plot Results ---
print("Plotting results...")

# Plot 1: Predictions vs Actual
plt.figure(figsize=(14, 7))
plot_index = df_features.index[-len(actual):]
plt.plot(plot_index, actual, label='Actual Close Price', color='blue', alpha=0.7)
plt.plot(plot_index, predictions, label='Predicted Close Price', color='red', linestyle='--')
plt.title(f'{FILENAME.split("_")[0]} Close Price Prediction (LSTM with SHAP Features, StandardScaler)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE)
print(f"Prediction plot saved as {PREDICTION_PLOT_FILE}")
plt.close() # Close the figure to free memory

# Plot 2: Confusion Matrix
print(f"Plotting confusion matrix...")
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Down/Same', 'Predicted Up'],
    yticklabels=['Actual Down/Same', 'Actual Up']
)
plt.title('Confusion Matrix - Price Direction Prediction')
plt.ylabel('Actual Direction')
plt.xlabel('Predicted Direction')
plt.tight_layout()
plt.savefig(CM_PLOT_FILE)
print(f"Confusion matrix plot saved as {CM_PLOT_FILE}")
plt.close()


print("\n--- LSTM Script Finished ---")