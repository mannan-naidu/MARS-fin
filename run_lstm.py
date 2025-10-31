import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress TensorFlow GPU logging and UserWarnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def create_dataset(dataset, time_step=60):
    """
    Converts a time-series array into input sequences and corresponding outputs.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def run_lstm_analysis(file_path, logs_dir):
    """
    Loads stock data, trains an LSTM model, saves detailed training logs,
    and returns the key performance metrics.
    """
    file_name = os.path.basename(file_path)
    stock_symbol = file_name.split('.')[0]
    print(f"--- Processing {file_name} ---")

    try:
        # --- 1. Data Loading and Preparation ---
        df = pd.read_csv(file_path, skiprows=[1])
        df_close = df.reset_index()['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

        training_size = int(len(scaled_data) * 0.80)
        train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]

        time_step = 60
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"WARNING: Not enough data for {file_name} to create sequences. Skipping.")
            return None

        # --- 2. Build the LSTM Model (Updated to remove UserWarning) ---
        model = Sequential([
            Input(shape=(time_step, 1)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')

        # --- 3. Train the Model and Log History ---
        print(f"Starting training for {stock_symbol}...")
        start_time = time.time()
        # Using verbose=1 to show the training progress for each epoch
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=1)
        end_time = time.time()
        
        # Save the training history to a CSV file
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = history.epoch
        log_path = os.path.join(logs_dir, f"{stock_symbol}_log.csv")
        history_df.to_csv(log_path, index=False)
        
        print(f"\nTraining for {stock_symbol} completed in {end_time - start_time:.2f} seconds.")
        print(f"Detailed training log saved to: {log_path}")

        # --- 4. Make Predictions and Evaluate ---
        test_predict = model.predict(X_test, verbose=0)
        test_predict_unscaled = scaler.inverse_transform(test_predict)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate metrics
        mae = mean_absolute_error(y_test_unscaled, test_predict_unscaled)
        mse = mean_squared_error(y_test_unscaled, test_predict_unscaled)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_unscaled, test_predict_unscaled)

        print("\n--- Final Metrics ---")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R-squared: {r2:.4f}")
        print("-" * 40 + "\n")

        return stock_symbol, mae, mse, rmse, r2

    except Exception as e:
        print(f"\nAn error occurred while processing {file_path}: {e}")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    # The folder where your CSV files are located
    data_folder = "Nifty50_CSVs"
    
    # Create a new directory for the logs if it doesn't exist
    logs_folder = "lstm_training_logs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    results_folder = os.getcwd()
    all_metrics = []

    if not os.path.isdir(data_folder):
        print(f"Error: The data folder '{data_folder}' was not found.")
    else:
        files_to_process = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if not files_to_process:
            print(f"No CSV files found in '{data_folder}'.")
        else:
            print(f"Found {len(files_to_process)} CSV files. Starting LSTM analysis...\n")
            for stock_file in files_to_process:
                full_path = os.path.join(data_folder, stock_file)
                result = run_lstm_analysis(full_path, logs_folder)
                if result:
                    all_metrics.append(result)

            if all_metrics:
                results_df = pd.DataFrame(all_metrics, columns=['Stock_Symbol', 'MAE', 'MSE', 'RMSE', 'R_Squared'])
                results_df = results_df.sort_values(by='RMSE', ascending=True)
                output_csv_path = os.path.join(results_folder, 'lstm_model_metrics.csv')
                results_df.to_csv(output_csv_path, index=False)
                print(f"\nAll final model metrics have been saved to: {output_csv_path}")

            print("\nProcess finished for all files.")