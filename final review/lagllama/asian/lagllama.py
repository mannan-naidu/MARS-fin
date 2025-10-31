import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForTimeSeriesPrediction, AutoProcessor, TrainingArguments, Trainer
from datasets import Dataset
from evaluate import load as load_metric
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import traceback
from functools import partial

print("Starting Lag-Llama Script for Asian Paints (SHAP Features)...")

# --- Configuration ---
FILENAME = 'ASIANPAINTS_master_data.csv'
TARGET_COLUMN = 'Close'
DATE_COLUMN = 'date'
LAG_LLAMA_MODEL_NAME = "Salesforce/Lag-Llama" # <<< REPLACE with actual model name
OUTPUT_DIR = "./lagllama_asianpaints_results"
LOGGING_DIR = './logs_asianpaints'
OUTPUT_PLOT_FILENAME = 'lagllama_predictions_asianpaints.png'

CONTEXT_LENGTH = 128
PREDICTION_LENGTH = 5
TEST_SIZE = 0.2
TRAIN_EPOCHS = 3
BATCH_SIZE = 16

# --- Feature Selection (Based on Asian Paints SHAP plot) ---
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', # Price/volume
    'Crude_Oil_Brent',                       # Macro
    'USD_INR_Exchange_Rate',                 # Macro
    '10Y_Bond_Yield',                        # Macro
    'sentiment_weighted_mean',               # Sentiment
    'news_count',                            # Sentiment
    'Policy_Repo_Rate',                      # Macro
    # Include one key quarterly metric if desired, though lower importance
    'Net_Profit_Loss_For_the_Period'
]
COVARIATE_FEATURES = [f for f in FEATURES if f != TARGET_COLUMN]

# --- Helper Function to Create Sequences ---
def create_sequences(data, context_length, prediction_length, target_col, covariate_cols):
    indices = []
    start_times = []
    target_data = data[[target_col]].to_numpy()
    covariate_data = data[covariate_cols].to_numpy() if covariate_cols else None
    times = data.index.to_numpy()
    total_len = len(data)
    for i in range(total_len - context_length - prediction_length + 1):
        indices.append(i)
        start_times.append(times[i])
    sequences_dict = {
        'start': start_times,
        'past_values': [target_data[i:i+context_length].flatten().tolist() for i in indices],
        'past_time_features': [np.arange(context_length).tolist()] * len(indices), # Example
        'past_observed_mask': [np.ones(context_length).tolist()] * len(indices),
        'future_values': [target_data[i+context_length:i+context_length+prediction_length].flatten().tolist() for i in indices],
        'future_time_features': [np.arange(context_length, context_length + prediction_length).tolist()] * len(indices), # Example
    }
    if covariate_data is not None:
        sequences_dict['past_feat_dynamic_real'] = [covariate_data[i:i+context_length].tolist() for i in indices]
        sequences_dict['future_feat_dynamic_real'] = [covariate_data[i+context_length:i+context_length+prediction_length].tolist() for i in indices]
    return Dataset.from_dict(sequences_dict)

# --- 1. Load Data ---
print(f"Loading data from {FILENAME}...")
if not os.path.exists(FILENAME): print(f"--- ERROR --- File not found: {FILENAME}"); exit()
try:
    df = pd.read_csv(FILENAME)
    # Use dayfirst=True based on previous LSTM error
    df[DATE_COLUMN] = pd.to_datetime(df.pop(DATE_COLUMN), dayfirst=True, errors='coerce')
    df = df.dropna(subset=[DATE_COLUMN])
    df = df.set_index(DATE_COLUMN).sort_index()
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"--- ERROR --- Features missing: {missing_features}")
        print(f"Available: {df.columns.tolist()}")
        exit()
    df_features = df[FEATURES].copy()
    for col in FEATURES: df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    df_features = df_features.dropna()
    print(f"Data loaded. Shape: {df_features.shape}")
except Exception as e: print(f"--- ERROR Loading Data --- \n{e}"); exit()

# --- 2. Split Data ---
split_index = int(len(df_features) * (1 - TEST_SIZE))
train_df = df_features.iloc[:split_index]
test_df = df_features.iloc[split_index:]
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- 3. Scale Data ---
print("Scaling data (StandardScaler)...")
target_scaler = StandardScaler()
covariate_scaler = StandardScaler()
train_df[TARGET_COLUMN] = target_scaler.fit_transform(train_df[[TARGET_COLUMN]])
if COVARIATE_FEATURES: train_df[COVARIATE_FEATURES] = covariate_scaler.fit_transform(train_df[COVARIATE_FEATURES])
test_df_scaled = df_features.iloc[split_index - CONTEXT_LENGTH:].copy() # Include overlap
test_df_scaled[TARGET_COLUMN] = target_scaler.transform(test_df_scaled[[TARGET_COLUMN]])
if COVARIATE_FEATURES: test_df_scaled[COVARIATE_FEATURES] = covariate_scaler.transform(test_df_scaled[COVARIATE_FEATURES])
print("Data scaled.")

# --- 4. Create Hugging Face Datasets ---
print("Creating Hugging Face Datasets...")
train_dataset = create_sequences(train_df, CONTEXT_LENGTH, PREDICTION_LENGTH, TARGET_COLUMN, COVARIATE_FEATURES)
val_cutoff = int(len(train_dataset) * 0.9)
val_dataset = train_dataset.select(range(val_cutoff, len(train_dataset)))
train_dataset = train_dataset.select(range(val_cutoff))

# Test dataset for prediction generation
test_indices = []
test_times = test_df_scaled.index.to_numpy()
test_target_data = test_df_scaled[[TARGET_COLUMN]].to_numpy()
test_covariate_data = test_df_scaled[COVARIATE_FEATURES].to_numpy() if COVARIATE_FEATURES else None
for i in range(len(test_df_scaled) - CONTEXT_LENGTH + 1): test_indices.append(i)
test_sequences_dict = {
     'start': [test_times[i] for i in test_indices],
     'past_values': [test_target_data[i:i+CONTEXT_LENGTH].flatten().tolist() for i in test_indices],
     'past_time_features': [np.arange(CONTEXT_LENGTH).tolist()] * len(test_indices),
     'past_observed_mask': [np.ones(CONTEXT_LENGTH).tolist()] * len(test_indices),
}
if COVARIATE_FEATURES:
     test_sequences_dict['past_feat_dynamic_real'] = [test_covariate_data[i:i+CONTEXT_LENGTH].tolist() for i in test_indices]
     test_sequences_dict['future_feat_dynamic_real'] = [test_covariate_data[i+CONTEXT_LENGTH:min(i+CONTEXT_LENGTH+PREDICTION_LENGTH, len(test_df_scaled))].tolist() for i in test_indices]
     # Pad the last sequence if needed
     last_len = len(test_sequences_dict['future_feat_dynamic_real'][-1])
     if last_len < PREDICTION_LENGTH:
         padding = np.zeros((PREDICTION_LENGTH - last_len, len(COVARIATE_FEATURES))).tolist()
         test_sequences_dict['future_feat_dynamic_real'][-1].extend(padding)

test_dataset_predict = Dataset.from_dict(test_sequences_dict)
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test pred size: {len(test_dataset_predict)}")

# --- 5. Load Model & Processor ---
print(f"Loading model and processor: {LAG_LLAMA_MODEL_NAME}...")
try:
    # Use AutoTimeSeriesProcessor - this often handles feature extraction/tokenization
    processor = AutoProcessor.from_pretrained(LAG_LLAMA_MODEL_NAME)
    # Use AutoModelForTimeSeriesPrediction - common class for these models
    model = AutoModelForTimeSeriesPrediction.from_pretrained(LAG_LLAMA_MODEL_NAME)
except Exception as e: print(f"--- ERROR Loading Model --- \n{e}"); exit()

# --- Preprocessing Function ---
def transform_data(batch, processor):
    # Pass data to the processor; it should handle creation of necessary inputs
    inputs = processor(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        future_values=batch.get("future_values"), # Only present during training/eval
        future_time_features=batch.get("future_time_features"), # Only present during training/eval
        past_feat_dynamic_real=batch.get("past_feat_dynamic_real"),
        future_feat_dynamic_real=batch.get("future_feat_dynamic_real"),
        return_tensors="pt"
    )
    # The processor *might* put future_values into 'labels' automatically, check docs
    if "future_values" in batch and "labels" not in inputs:
         inputs["labels"] = inputs.pop("future_values") # Manually set labels if needed
    return inputs

train_dataset.set_transform(partial(transform_data, processor=processor))
val_dataset.set_transform(partial(transform_data, processor=processor))

# --- 6. Define Training Arguments & Trainer ---
print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=TRAIN_EPOCHS, save_strategy="epoch",
    logging_dir=LOGGING_DIR, logging_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="eval_loss", # Use loss
    greater_is_better=False, report_to="none"
)
# Define simple MSE metric for eval loss
def compute_metrics(eval_pred):
    # Predictions might be tuples (e.g., logits, loss), extract necessary parts
    # Assuming predictions are directly comparable to labels after processing
    predictions, labels = eval_pred
    # The model might output predictions for the whole sequence (past+future)
    # Ensure you compare only the future part
    pred_future = predictions[:, -PREDICTION_LENGTH:]
    labels_future = labels[:, -PREDICTION_LENGTH:]
    mse = np.mean((pred_future - labels_future) ** 2)
    return {"mse": mse}

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  eval_dataset=val_dataset, compute_metrics=compute_metrics)

# --- 7. Train Model ---
print("Training model...")
try: trainer.train(); print("Training finished.")
except Exception as e: print(f"--- ERROR Training --- \n{e}"); traceback.print_exc(); exit()

# --- 8. Make Predictions ---
print("Making predictions...")
all_forecasts = []
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for i in range(0, len(test_dataset_predict), BATCH_SIZE):
        batch_indices = range(i, min(i + BATCH_SIZE, len(test_dataset_predict)))
        raw_batch = test_dataset_predict[batch_indices] # Get raw data dict

        # Prepare inputs specifically for model.generate()
        inputs = processor(
            past_values=raw_batch["past_values"],
            past_time_features=raw_batch["past_time_features"],
            past_observed_mask=raw_batch["past_observed_mask"],
            past_feat_dynamic_real=raw_batch.get("past_feat_dynamic_real"),
            future_feat_dynamic_real=raw_batch.get("future_feat_dynamic_real"), # Provide future covariates
            return_tensors="pt"
        ).to(device)

        # Generate predictions
        outputs = model.generate(
            **inputs,
            prediction_length=PREDICTION_LENGTH
        )
        # Assuming output shape: (batch_size, num_samples, prediction_length)
        # Extract mean prediction: shape (batch_size, prediction_length)
        mean_forecast = outputs.sequences.mean(dim=1)
        all_forecasts.append(mean_forecast.cpu().numpy())

predictions_scaled = np.concatenate(all_forecasts, axis=0)
num_predictions = predictions_scaled.shape[0]
print(f"Generated {num_predictions} forecast sequences.")

# --- 9. Inverse Scale & Evaluate ---
print("Inverse scaling and evaluating...")
actual_unscaled = test_df[TARGET_COLUMN].iloc[CONTEXT_LENGTH : CONTEXT_LENGTH + num_predictions].values
pred_first_step_scaled = predictions_scaled[:, 0].reshape(-1, 1) # First step prediction
pred_first_step_unscaled = target_scaler.inverse_transform(pred_first_step_scaled).flatten()

eval_len = min(len(pred_first_step_unscaled), len(actual_unscaled))
if eval_len == 0: print("--- ERROR --- No overlapping predictions/actuals."); exit()

mae = np.mean(np.abs(pred_first_step_unscaled[:eval_len] - actual_unscaled[:eval_len]))
rmse = np.sqrt(np.mean((pred_first_step_unscaled[:eval_len] - actual_unscaled[:eval_len])**2))
print(f"Test MAE (1st step): {mae:.4f}")
print(f"Test RMSE (1st step): {rmse:.4f}")

# --- 10. Plot Results ---
print("Plotting results...")
plt.figure(figsize=(14, 7))
plot_index = test_df.index[CONTEXT_LENGTH : CONTEXT_LENGTH + eval_len]
plt.plot(plot_index, actual_unscaled[:eval_len], label='Actual Close Price', color='blue', alpha=0.7)
plt.plot(plot_index, pred_first_step_unscaled[:eval_len], label=f'LagLlama Predicted (Step 1)', color='red', linestyle='--')
plt.title(f'{FILENAME.split("_")[0]} Close Price Prediction using LagLlama (SHAP Features)')
plt.xlabel('Date'); plt.ylabel('Close Price'); plt.legend(); plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILENAME)
print(f"Plot saved as {OUTPUT_PLOT_FILENAME}")

print("\n--- LagLlama Script Finished (Conceptual) ---")