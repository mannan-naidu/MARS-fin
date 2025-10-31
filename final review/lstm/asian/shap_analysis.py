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
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import shap

print("Starting SHAP Analysis for 'Events Only' Multi-Output Delta Model...")

# --- 1. Configuration & Tuning ---
FILENAME = 'ASIANPAINTS_master_data_engineered.csv'
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
    "LOSS_WEIGHT_PRICE": 1.0,
    "LOSS_WEIGHT_DIR": 0.5
}
MODEL_ARCH_NAME = "GRU" if TUNING_VARS["USE_GRU"] else "LSTM"

RESULTS_FILE = f'{STOCK_TICKER}_multi_output_delta_{MODEL_ARCH_NAME}_events_only.json'
MODEL_SUMMARY_FILE = f'{STOCK_TICKER}_multi_output_delta_{MODEL_ARCH_NAME}_events_only_model_summary.txt'
SHAP_PLOT_FILE = f'{STOCK_TICKER}_shap_feature_importance_bar_plot.png'

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

# --- 3. Feature Selection (Events Only) ---
print("Defining 'Events Only' features...")
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'sentiment_weighted_mean', 'news_count',
]
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
print(f"Using {len(FEATURES)} 'Events Only' features: {FEATURES}")

# --- 4. Preprocessing & Target Creation ---
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
all_data = all_data.dropna()

if all_data.empty:
    print("--- ERROR --- DataFrame empty after feature/target creation.")
    exit()

df_features_clean = all_data[FEATURES]
df_target_reg_clean = all_data[[TARGET_REG_COL]]
df_target_class_clean = all_data[[TARGET_CLASS_COL]]
print(f"Data cleaned. Feature Shape: {df_features_clean.shape}, Target Shape: {df_target_reg_clean.shape}")

# --- 5. Scaling ---
print("Scaling data (MinMaxScaler for X, StandardScaler for y_delta)...")
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_features_clean)
scaler_y_reg = StandardScaler()
y_reg_scaled = scaler_y_reg.fit_transform(df_target_reg_clean)
y_class = df_target_class_clean.values

# --- 6. Create Sequences ---
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

# --- 7. Split Data ---
print(f"Splitting data (Test size: {TUNING_VARS['TEST_SIZE']})...")
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X_seq, y_reg_seq, y_class_seq, 
    test_size=TUNING_VARS['TEST_SIZE'], 
    shuffle=False
)
y_train_dict = {'price': y_reg_train, 'direction': y_class_train}
y_test_dict = {'price': y_reg_test, 'direction': y_class_test}
print(f"Train shapes: X={X_train.shape}, Test shapes: X={X_test.shape}")

# --- 8. Build Model ---
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

# --- 9. Train Model ---
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

# --- 10. SHAP Analysis ---
print("\n--- Starting SHAP Analysis ---")
print("Please ensure 'shap' is installed: pip install shap")

try:
    # ### <<< MODIFICATION 1: Create a wrapper model for the direction output >>> ###
    # This new model reuses the layers and weights from the trained model
    # but only has the single output we want to explain (index [1] = direction).
    model_direction_only = Model(inputs=model.inputs, outputs=model.outputs[1])

    # 1. Create a background dataset
    background_data = shap.sample(X_train, 100)

    # 2. Initialize the DeepExplainer
    # ### <<< MODIFICATION 2: Pass the NEW, single-output model to the explainer >>> ###
    explainer = shap.DeepExplainer(model_direction_only, background_data)

    # 3. Get SHAP values for the test set
    print("Calculating SHAP values... (this may take a few minutes)")
    # This will now return a list with *one* item, not two.
    shap_values = explainer.shap_values(X_test)

    # 4. Get the single SHAP array
    # ### <<< MODIFICATION 3: Get the first (and only) item from the list >>> ###
    shap_values_direction = shap_values[0]
    
    # 5. Create a Global Feature Importance Plot
    mean_abs_shap_values = np.mean(np.abs(shap_values_direction), axis=(0, 1))
    shap_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': mean_abs_shap_values
    }).sort_values(by='importance', ascending=False)
    
    print("\nGlobal Feature Importance (Mean Absolute SHAP):")
    print(shap_df)

    # 6. Plot and Save the Bar Chart
    print(f"Saving SHAP feature importance plot to {SHAP_PLOT_FILE}...")
    
    # We average over the time axis for the summary plot
    shap_values_avg_over_time = np.mean(shap_values_direction, axis=1)

    plt.figure()
    shap.summary_plot(
        shap_values_avg_over_time, 
        X_test.mean(axis=1), 
        feature_names=FEATURES,
        plot_type="bar",
        show=False
    )
    plt.title(f"{STOCK_TICKER} - SHAP Feature Importance for Direction")
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_FILE)
    print(f"SHAP plot saved as {SHAP_PLOT_FILE}")
    plt.close()
    
    # 7. Also save the text data
    shap_df.to_csv(f"{STOCK_TICKER}_shap_feature_importance.csv", index=False)
    print(f"SHAP importance values saved to {STOCK_TICKER}_shap_feature_importance.csv")

except Exception as e:
    print(f"\n--- ERROR during SHAP Analysis ---: {e}")
    print("This often happens if the 'shap' library is not installed or has a version conflict.")
    traceback.print_exc()

print("\n--- SHAP Analysis Script Finished ---")