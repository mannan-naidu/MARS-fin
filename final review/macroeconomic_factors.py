import requests
import pandas as pd
from io import StringIO
import time # To add delays between requests if needed
import logging # Import the logging library

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, # Set the minimum level of messages to log
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration ---
# !!! REPLACE "YOUR_API_KEY" WITH YOUR ACTUAL FRED API KEY !!!
FRED_API_KEY = "9cf527c92afcd7fe1dfd880cc6e79380"

# Define the start and end dates
START_DATE_STR = "2015-01-01"
END_DATE_STR = "2025-12-31" # Use the exact end date
# Convert string dates to datetime objects for comparison later
START_DATE = pd.to_datetime(START_DATE_STR)
END_DATE = pd.to_datetime(END_DATE_STR)

# Define the FRED Series IDs for the Indian macro features
SERIES_IDS = {
    "Policy_Repo_Rate": "INTDSRINM193N", # Monthly
    "10Y_Bond_Yield": "INDIRLTLT01STM", # Monthly
    "Call_Money_Rate": "IRSTCI01INM156N", # Monthly
    "CPI_Index": "INDCPIALLMINMEI", # Monthly
    "IIP_Index": "INDPROINDMISMEI", # Monthly
    "GDP_Growth_QoQ": "NGDPNSAXDCINQ", # Quarterly
    "USD_INR_Exchange_Rate": "DEXINUS", # Daily
    "Trade_Exports_USD": "XTEXVA01INM667S", # Monthly
    "Trade_Imports_USD": "XTIMVA01INM667S", # Monthly
    "FX_Reserves": "TRESEGINM052N", # Monthly
    "Crude_Oil_Brent": "DCOILBRENTEU", # Daily
    "Crude_Oil_WTI": "DCOILWTICO", # Daily
    "Gold_Price_London": "GOLDPMGBD228NLBM", # Daily
    # Add other IDs here if needed
    # --- Example of a potentially problematic ID for testing ---
    # "Example_Discontinued": "DISCONTINUEDSERIESID",
}

# Base URL for FRED API series observations endpoint
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Dictionary to store the resulting DataFrames
all_data = {}

# --- Fetching Loop ---
logging.info("Starting data fetch from FRED...")

for name, series_id in SERIES_IDS.items():
    logging.info(f"Attempting to fetch: {name} ({series_id})")

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json", # Request JSON format
        "observation_start": START_DATE_STR,
        "observation_end": END_DATE_STR,
        "sort_order": "asc", # Get data in chronological order
    }

    try:
        response = requests.get(FRED_API_URL, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            logging.warning(f"No data returned for {name} ({series_id}) within the specified period.")
            continue # Skip to the next series

        # Convert observations to DataFrame
        df = pd.DataFrame(observations)

        # Keep only date and value, rename columns
        df = df[['date', 'value']]
        df.rename(columns={'value': name}, inplace=True)

        # Convert date to datetime objects
        df['date'] = pd.to_datetime(df['date'])

        # Convert value to numeric, coercing errors (like FRED's '.') to NaN
        df[name] = pd.to_numeric(df[name], errors='coerce')

        # Set date as index
        df.set_index('date', inplace=True)

        # --- Data Validation and Logging ---
        fetched_start_date = df.index.min()
        fetched_end_date = df.index.max()
        data_point_count = len(df)
        nan_count = df[name].isna().sum()

        log_message = (
            f"Successfully fetched {name} ({series_id}). "
            f"Fetched {data_point_count} data points. "
            f"NaN values: {nan_count}/{data_point_count}. "
            f"Fetched range: {fetched_start_date.strftime('%Y-%m-%d')} to {fetched_end_date.strftime('%Y-%m-%d')}."
        )

        # Check if the fetched date range covers the requested range
        if fetched_start_date > START_DATE or fetched_end_date < END_DATE:
            logging.warning(f"{log_message} - WARNING: Does not fully cover requested range {START_DATE_STR} to {END_DATE_STR}.")
        else:
            logging.info(log_message)

        all_data[name] = df

        # Optional: Add a small delay
        # time.sleep(0.5)

    except requests.exceptions.HTTPError as e:
        # Specifically log HTTP errors (like 404 Not Found, 400 Bad Request)
        logging.error(f"HTTP Error fetching {name} ({series_id}): {e.response.status_code} - {e.response.reason}. URL: {e.request.url}")
        # Log the response text if available for more details (e.g., API error message)
        if e.response.text:
            logging.error(f"FRED API Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        # Log other network/request related errors
        logging.error(f"Request Error fetching {name} ({series_id}): {e}")
    except Exception as e:
        # Log any other unexpected errors during processing
        logging.error(f"Unexpected Error processing {name} ({series_id}): {e}")

logging.info("--- Fetch process complete ---")

# --- Optional: Combine into a single DataFrame ---
if all_data:
    logging.info("Combining fetched data into a single DataFrame...")
    combined_df = pd.DataFrame()
    for name, df in all_data.items():
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')

    combined_df.sort_index(inplace=True)

    logging.info("--- Combined DataFrame Head (first 5 rows) ---")
    print(combined_df.head().to_string()) # Use to_string to prevent truncation
    logging.info("--- Combined DataFrame Tail (last 5 rows) ---")
    print(combined_df.tail().to_string())
    logging.info("--- Combined DataFrame Info ---")
    combined_df.info() # This will show non-null counts per column

    # --- Optional: Save to CSV ---
    combined_df.to_csv("macro_data_fred_detailed.csv")
    logging.info("Combined data saved to indian_macro_data_fred_detailed.csv")
else:
    logging.warning("No data was successfully fetched to combine.")