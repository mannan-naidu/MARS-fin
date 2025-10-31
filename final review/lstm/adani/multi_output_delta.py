import pandas as pd
import numpy as np
import pandas_ta as ta # Make sure you have run: pip install pandas_ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import traceback
import json
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

print("Starting LSTM/GRU Multi-Output (Price CHANGE/DELTA) Model - EVENTS + TA FEATURES...")

# --- 1. Configuration & Tuning ---
FILENAME = 'ADANIENT_master_data_engineered.csv' # <<< Back to ADANIENT
TARGET_COLUMN = 'Close'
STOCK_TICKER = FILENAME.split('_')[0]

# --- Hyperparameters ---
TUNING_VARS = {
    "SEQUENCE_LENGTH": 30,
    "LSTM_UNITS": 100,
    "DROPOUT": 0.2,
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "TEST_SIZE": 0.2,
    "USE_GRU": True, # True to use GRU, False to use LSTM
    "LOSS_WEIGHT_PRICE": 1.0,
    "LOSS_WEIGHT_DIR": 0.5
}

MODEL_ARCH_NAME = "GRU" if TUNING_VARS["USE_GRU"] else "LSTM"

# --- Output filenames ---
RESULTS_FILE = f'{STOCK_TICKER}_multi_output_delta_{MODEL_ARCH_NAME}_events_plus_ta.json'
MODEL_SUMMARY_FILE = f'{STOCK_TICKER}_multi_output_delta_{MODEL_ARCH_NAME}_events_plus_ta_model_summary.txt'
PREDICTION_PLOT_FILE = f'{STOCK_TICKER}_multi_output_delta_{MODEL_ARCH_NAME}_events_plus_ta_predictions.png'
CM_PLOT_FILE = f'{STOCK_TICKER}_multi_output_delta_{MODEL_ARCH_NAME}_events_plus_ta_confusion_matrix.png'


# --- 2. Load Data ---
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
    print(f"--- ERROR Loading Data ---: {e}")
    exit()

# --- 3. Add Technical Analysis (TA) Features ---
print("Calculating Technical Analysis features...")
try:
    # 1. RSI (Momentum)
    df['TA_RSI_14'] = ta.rsi(df['Close'], length=14)
    
    # 2. MACD (Trend)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['TA_MACD_hist'] = macd['MACDh_12_26_9'] # The histogram (MACD line - signal line)

    # 3. Bollinger Bands (Volatility)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['TA_BB_pctB'] = bbands['BBP_20_2.0'] # %B (percent B)
    df['TA_BB_width'] = bbands['BBB_20_2.0'] # Bandwidth

    TA_FEATURES = ['TA_RSI_14', 'TA_MACD_hist', 'TA_BB_pctB', 'TA_BB_width']
    print(f"TA features calculated: {TA_FEATURES}")

except Exception as e:
    print(f"--- ERROR calculating TA features ---: {e}")
    print("Please ensure 'pandas_ta' is installed: pip install pandas_ta")
    TA_FEATURES = []


# --- 4. Feature Selection (Events Only + TA) ---
print("Defining 'Events Only + TA' features...")
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'sentiment_weighted_mean', 'news_count',
]
FEATURES.extend(TA_FEATURES)

BASE_SHAP_FEATURES = [
    'Crude_Oil_Brent', 'USD_INR_Exchange_Rate', '10Y_Bond_Yield',
    'Policy_Repo_Rate', 'Net_Profit_Loss_For_the_Period'
]
available_cols = df.columns.tolist()

for base_name in BASE_SHAP_FEATURES:
    chg_col = f"{base_name}_abs_change"
    stale_col = f"{base_name}_days_since_change"
    if (chg_col in available_cols and stale_col in available_cols):
        FEATURES.extend([chg_col, stale_col])
    else:
        print(f"--- WARNING --- Skipping '{base_name}' as its 2 event components are not fully available.")

FEATURES = sorted(list(set(FEATURES)))
print(f"Using {len(FEATURES)} 'Events + TA' features: {FEATURES}") # Should be 17 + 4 = 21 features

# --- 5. Preprocessing & Target Creation ---
print("Preprocessing data and creating targets (Price Change and Direction)...")
TARGET_REG_COL = 'Close_Change'
TARGET_CLASS_COL = 'Direction'

df_target_reg = df[[TARGET_COLUMN]].diff().copy()
df_target_reg.columns = [TARGET_REG_COL] 
df_target_class = (df_target_reg[TARGET_REG_COL] > 0).astype(int).to_frame()
df_target_class.columns = [TARGET_CLASS_COL] 
df_features = df[FEATURES].copy()

print("Coercing features to numeric...")
for col in df_features.columns:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

all_data = pd.concat([df_features, df_target_reg, df_target_class], axis=1)
# .dropna() removes NaNs from .diff() and from the TA indicators (e.g., first 14 days for RSI)
all_data = all_data.dropna()

if all_data.empty:
    print("--- ERROR --- DataFrame empty after feature/target creation and NaN drop.")
    exit()

df_features_clean = all_data[FEATURES]
df_target_reg_clean = all_data[[TARGET_REG_COL]]
df_target_class_clean = all_data[[TARGET_CLASS_COL]]

print(f"Data cleaned. Feature Shape: {df_features_clean.shape}, Target Shape: {df_target_reg_clean.shape}")

# --- 6. Scaling ---
print("Scaling data (MinMaxScaler for X, StandardScaler for y_delta)...")
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_features_clean)

scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(df_target_reg_clean)
y_class = df_target_class_clean.values

# --- 7. Create Sequences ---
print(f"Creating sequences (length {TUNING_VARS['SEQUENCE_LENGTH']})...")
X_seq, y_reg_seq, y_class_seq = [], [], []
for i in range(TUNING_VARS['SEQUENCE_LENGTH'], len(X_scaled)):
    X_seq.append(X_scaled[i-TUNING_VARS['SEQUENCE_LENGTH']:i])
    y_reg_seq.append(y_reg_scaled[i])
    y_class_seq.append(y_class[i])

X_seq, y_reg_seq, y_class_seq = np.array(X_seq), np.array(y_reg_seq), np.array(y_class_seq)

if X_seq.shape[0] == 0:
    print(f"--- ERROR --- Not enough data for sequences.")
    exit()
print(f"Sequences created. X shape: {X_seq.shape}")

# --- 8. Split Data ---
print(f"Splitting data (Test size: {TUNING_VARS['TEST_SIZE']})...")
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X_seq, y_reg_seq, y_class_seq, 
    test_size=TUNING_VARS['TEST_SIZE'], 
    shuffle=False
)

y_train_dict = {'price': y_reg_train, 'direction': y_class_train}
y_test_dict = {'price': y_reg_test, 'direction': y_class_test}

print(f"Train shapes: X={X_train.shape}, y_reg={y_reg_train.shape}, y_class={y_class_train.shape}")
print(f"Test shapes: X={X_test.shape}, y_reg={y_reg_test.shape}, y_class={y_class_test.shape}")

# --- 9. Build Model ---
print(f"Building Multi-Output {MODEL_ARCH_NAME} model...")
RecurrentLayer = GRU if TUNING_VARS["USE_GRU"] else LSTM

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]), name='input')
shared_base = RecurrentLayer(TUNING_VARS['LSTM_UNITS'], return_sequences=True)(input_layer)
shared_base = Dropout(TUNING_VARS['DROPOUT'])(shared_base)
shared_base = RecurrentLayer(TUNING_VARS['LSTM_UNITS'], return_sequences=False)(shared_base)
shared_base = Dropout(TUNING_VARS['DROPOUT'])(shared_base)

price_branch = Dense(25, activation='relu')(shared_base)
price_output = Dense(1, activation='linear', name='price')(price_branch)
direction_branch = Dense(25, activation='relu')(shared_base)
direction_output = Dense(1, activation='sigmoid', name='direction')(direction_branch)

model = Model(inputs=input_layer, outputs=[price_output, direction_output])

model.compile(
    optimizer='adam',
    loss={'price': 'mean_squared_error', 'direction': 'binary_crossentropy'},
    loss_weights={'price': TUNING_VARS['LOSS_WEIGHT_PRICE'], 'direction': TUNING_VARS['LOSS_WEIGHT_DIR']},
    metrics={'price': 'mae', 'direction': 'accuracy'}
)
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
    y_train_dict,
    batch_size=TUNING_VARS['BATCH_SIZE'],
    epochs=TUNING_VARS['EPOCHS'],
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("Training finished.")

# --- 11. Evaluate Model ---
print("Evaluating model...")
losses = model.evaluate(X_test, y_test_dict, verbose=0)
print(f"Total Test Loss: {losses[0]:.4f}")
print(f"Price (MSE) Test Loss: {losses[1]:.4f}")
print(f"Direction (BCE) Test Loss: {losses[2]:.4f}")
print(f"Price (MAE) Test Metric: {losses[3]:.4f}")
print(f"Direction (Accuracy) Test Metric: {losses[4]:.4f}")

# --- 12. Make Predictions & Inverse Scale ---
print("Making predictions...")
predictions_list = model.predict(X_test)
predictions_scaled = predictions_list[0]
predictions_direction_raw = predictions_list[1]

predictions = scaler_y_reg.inverse_transform(predictions_scaled)
actual = scaler_y_reg.inverse_transform(y_reg_test)
predicted_direction = (predictions_direction_raw > 0.5).astype(int)
actual_direction = y_class_test.astype(int)

# --- 13. Calculate REGRESSION Metrics ---
print("\n--- Naive Baseline (Predicting 0 Change) ---")
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
    print("\n*** SUCCESS: Model REGRESSION RMSE is BETTER than the Naime Baseline. ***")
else:
    print("\n--- INFO: Model REGRESSION RMSE is WORSE than the Naive Baseline. ---")

# --- 14. Calculate CLASSIFICATION Metrics ---
print("\n--- Classification Metrics (Direction Prediction) ---")
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

print(f"Directional Accuracy: {accuracy:.4f} (from model head)")
print(f"Directional Precision (for 'Up'): {precision:.4f}")
print(f"Directional Recall (for 'Up'): {recall:.4f}")
print(f"Directional F1-Score (for 'Up'): {f1:.4f}")
print("Confusion Matrix (Directional):")
print(cm)
print("\nClassification Report (Directional):")
print(report)

# --- 1In. Save Metrics and Features to File ---
print(f"\nSaving results to {RESULTS_FILE}...")
results = {
    'model_name': f"{STOCK_TICKER}_MultiOutput_Delta_Events_TA",
    'tuning_params': TUNING_VARS,
    'model_architecture': MODEL_ARCH_NAME,
    'features_used': FEATURES,
    'feature_count': len(FEATURES),
    'evaluation_losses_scaled': {
        'total_loss': losses[0],
        'price_mse_loss': losses[1],
        'direction_bce_loss': losses[2],
        'price_mae_metric': losses[3],
        'direction_accuracy_metric': losses[4]
    },
    'regression_metrics_unscaled_delta': {
        'model_mae': mae,
        'model_rmse': rmse,
        'naive_baseline_mae': naive_mae,
        'naive_baseline_rmse': naive_rmse,
        'improvement_over_naive_rmse': naive_rmse - rmse
    },
    'classification_metrics_directional': {
        'accuracy': accuracy,
        'precision_up': precision,
        'recall_up': recall,
        'f1_score_up': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
}

try:
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
except Exception as e:
    print(f"--- ERROR --- Could not save results to JSON: {e}")

# --- 16. Plot Results ---
print("Plotting results...")
plt.figure(figsize=(14, 7))
plot_index = df_features_clean.index[-len(actual):]
plt.plot(plot_index, actual, label='Actual Price Change', color='blue', alpha=0.7)
plt.plot(plot_index, predictions, label='Predicted Price Change', color='red', linestyle='--', alpha=0.8)
plt.title(f'{STOCK_TICKER} Price CHANGE Prediction ({MODEL_ARCH_NAME} - Events + TA)')
plt.xlabel('Date')
plt.ylabel('Price Change (Delta)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE)
print(f"Prediction plot saved as {PREDICTION_PLOT_FILE}")
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Down/Same', 'Predicted Up'],
    yticklabels=['Actual Down/Same', 'Actual Up']
)
plt.title('Confusion Matrix - Price Direction Prediction Head (Events + TA)')
plt.ylabel('Actual Direction')
plt.xlabel('Predicted Direction')
plt.tight_layout()
plt.savefig(CM_PLOT_FILE)
print(f"Confusion matrix plot saved as {CM_PLOT_FILE}")
plt.close()

print(f"\n--- {MODEL_ARCH_NAME} Multi-Output Delta Model w/TA Script Finished ---")