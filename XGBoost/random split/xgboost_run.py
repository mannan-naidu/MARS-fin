import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import time

def predict_stock_with_xgboost_gpu(file_path, results_dir):
    """
    Loads stock data, trains a GPU-accelerated XGBoost model,
    saves a prediction plot, and returns the RMSE score.

    Args:
        file_path (str): The full path to the stock data CSV file.
        results_dir (str): The directory where the output plot will be saved.
        
    Returns:
        tuple: A tuple containing the stock symbol and its RMSE, or None if an error occurs.
    """
    file_name = os.path.basename(file_path)
    stock_symbol = file_name.split('.')[0]
    print(f"Processing {file_name}...")

    try:
        # Load the data, skipping the second header row
        df = pd.read_csv(file_path, skiprows=[1])

        # --- 1. Data Cleaning and Preparation ---
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df.dropna(inplace=True)

        # --- 2. Feature Engineering ---
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)
        
        if df.empty:
            print(f"  - WARNING: Not enough data for {file_name} after feature engineering. Skipping.")
            return None

        # --- 3. Model Training ---
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI']
        target = 'Close'
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the XGBoost Regressor to use the GPU (Updated Method)
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            device='cuda',  # Use the modern 'device' parameter for GPU training
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"  - Training completed in {end_time - start_time:.2f} seconds.")

        # --- 4. Evaluation ---
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")

        # --- 5. Visualization ---
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        test_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()
        
        plt.plot(test_data.index, test_data['Actual'], label='Actual Prices', color='blue', alpha=0.7)
        plt.plot(test_data.index, test_data['Predicted'], label='Predicted Prices', color='red', linestyle='--')
        
        plt.title(f'XGBoost Price Prediction for {stock_symbol}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot to the specified results directory
        output_filename = f"{stock_symbol}_prediction.png"
        output_path = os.path.join(results_dir, output_filename)
        plt.savefig(output_path)
        plt.close()
        
        print(f"  - Plot saved to {output_path}")
        print("-" * 40)
        
        return stock_symbol, rmse

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the folder where your CSV files are located
    data_folder = "stock_csvs"
    
    # The results will be saved in the same directory as the script
    results_folder = os.getcwd()
    
    # List to store the results
    rmse_results = []

    # Check if the data folder exists
    if not os.path.isdir(data_folder):
        print(f"Error: The data folder '{data_folder}' was not found.")
        print("Please create it and place your CSV files inside.")
    else:
        # Get a list of all CSV files in the data folder
        files_to_process = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        if not files_to_process:
            print(f"No CSV files found in the '{data_folder}' directory.")
        else:
            print(f"Found {len(files_to_process)} CSV files to process.\n")
            
            for stock_file in files_to_process:
                # Construct the full path to the CSV file
                full_path = os.path.join(data_folder, stock_file)
                result = predict_stock_with_xgboost_gpu(full_path, results_folder)
                
                # If the function ran successfully, add the result to our list
                if result:
                    rmse_results.append(result)
            
            # --- Save the results to a file ---
            if rmse_results:
                results_df = pd.DataFrame(rmse_results, columns=['Stock_Symbol', 'RMSE'])
                results_df = results_df.sort_values(by='RMSE', ascending=True) # Sort by best performance
                
                output_csv_path = os.path.join(results_folder, 'all_rmse_results.csv')
                results_df.to_csv(output_csv_path, index=False)
                
                print(f"\nAll RMSE results have been saved to: {output_csv_path}")
            
            print("\nProcess finished for all files.")