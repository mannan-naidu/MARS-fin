import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import sys
import warnings
from tqdm import tqdm  # Using tqdm for a nice progress bar

# --- Configuration ---
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
# --- MODIFICATION: Set the Excel file name ---
EXCEL_FILE_NAME = 'asianpaints_news.xlsx' 
OUTPUT_FILE = 'daily_news_features_finbert_asianpaints.csv'
ROLLING_WINDOWS = [3, 7, 14] # 3-day, 1-week, 2-week windows
BATCH_SIZE = 64  # Process 64 headlines at a time. Adjust based on your GPU/CPU memory.

# --- Model Name ---
MODEL_NAME = "yiyanghkust/finbert-tone"

print("Starting FinBERT news feature engineering script for Excel...")
warnings.filterwarnings('ignore')

# --- STEP 1: Load and Combine All Sheets from Excel ---
print(f"Loading all sheets from Excel file: {EXCEL_FILE_NAME}...")
try:
    # sheet_name=None reads all sheets into a dictionary
    # The keys are sheet names, the values are DataFrames
    # Pandas (with openpyxl) is often smart enough to find the
    # header row even if there are junk rows above it.
    all_sheets_dict = pd.read_excel(EXCEL_FILE_NAME, sheet_name=None)
except FileNotFoundError:
    print(f"Error: The file '{EXCEL_FILE_NAME}' was not found.")
    print("Please make sure the Excel file is in the same directory as the script.")
    sys.exit()
except ImportError:
    print("\n--- ERROR ---")
    print("Missing required library to read Excel files.")
    print("Please run: pip install openpyxl")
    sys.exit()
except Exception as e:
    print(f"Error reading Excel file: {e}")
    sys.exit()

if not all_sheets_dict:
    print("Error: No sheets were found in the Excel file.")
    sys.exit()
    
print(f"Found {len(all_sheets_dict)} sheets: {list(all_sheets_dict.keys())}")
print("Concatenating all sheets into one dataset...")

# Concat all DataFrames from all sheets into one big DataFrame
# We add 'ignore_index=True' to create a new clean index
all_headlines = pd.concat(all_sheets_dict.values(), ignore_index=True)

# --- This is the start of the *original* script's Step 2 ---
# We just need to check if the required columns are present *after* concatenation
if 'Date' not in all_headlines.columns or 'Headline' not in all_headlines.columns:
     print("\n--- ERROR ---")
     print("The combined data is missing 'Date' or 'Headline' columns.")
     print("This can happen if the sheets have inconsistent headers or junk rows.")
     print(f"Found columns: {all_headlines.columns.tolist()}")
     print("Please check your Excel file so all sheets have a header row with 'Date' and 'Headline'.")
     sys.exit()

# Select only the columns we need, just in case Excel imported extra ones
all_headlines = all_headlines[['Date', 'Headline']]

print(f"Loaded a total of {len(all_headlines)} headlines from all sheets.")
print("Preprocessing data (dropping duplicates, cleaning)...")

# --- Step 2: Pre-process and Prepare for Model ---
all_headlines['date'] = pd.to_datetime(all_headlines['Date'], errors='coerce')
all_headlines = all_headlines.dropna(subset=['date', 'Headline'])
all_headlines['Headline'] = all_headlines['Headline'].astype(str)
all_headlines = all_headlines.drop_duplicates(subset=['date', 'Headline'])
all_headlines = all_headlines.reset_index(drop=True)

print(f"Total unique headlines to process: {len(all_headlines)}")

# --- Step 3: Score Sentiment with FinBERT ---
print(f"\nLoading FinBERT model: '{MODEL_NAME}'...")

# Check for GPU
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"
print(f"Using device: {device_name} (This may take a while on CPU)")

try:
    sentiment_pipeline = pipeline(
        'sentiment-analysis', 
        model=MODEL_NAME, 
        device=device
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an internet connection and 'transformers' is installed.")
    sys.exit()

print("Model loaded. Starting sentiment scoring (this is the slowest step)...")

headlines_list = all_headlines['Headline'].tolist()
sentiment_scores = []

def map_sentiment(pred):
    """Converts FinBERT output to a single score."""
    label = pred['label']
    score = pred['score']
    if label == 'Negative':
        return -score
    elif label == 'Positive':
        return score
    else:  # 'Neutral'
        return 0.0

for i in tqdm(range(0, len(headlines_list), BATCH_SIZE), desc="Scoring Headlines"):
    batch = headlines_list[i : i + BATCH_SIZE]
    batch_truncated = [h[:512] for h in batch] 
    
    try:
        results = sentiment_pipeline(batch_truncated)
        sentiment_scores.extend([map_sentiment(res) for res in results])
    except Exception as e:
        print(f"\nError processing batch {i}-{i+BATCH_SIZE}: {e}")
        sentiment_scores.extend([0.0] * len(batch))

all_headlines['sentiment_score'] = sentiment_scores
print("Sentiment scoring complete.")

# --- Step 4: Create Magnitude and Weighted Sentiment ---
print("Creating magnitude and weighted sentiment features...")
all_headlines['magnitude'] = all_headlines['sentiment_score'].abs()
all_headlines['sentiment_x_magnitude'] = all_headlines['sentiment_score'] * all_headlines['magnitude']

# --- Step 5: Daily Aggregation ---
print("Aggregating headlines to daily features...")
daily_agg = all_headlines.groupby('date').agg({
    'sentiment_score': ['mean', 'std', 'max', 'min', 'count'],
    'magnitude': ['mean', 'sum'],
    'sentiment_x_magnitude': 'sum'
}).reset_index()

daily_agg.columns = [
    'date',
    'sentiment_mean', 'sentiment_std', 'sentiment_max', 'sentiment_min', 'news_count',
    'magnitude_mean', 'magnitude_sum',
    'sentiment_x_magnitude_sum'
]

daily_agg['sentiment_weighted_mean'] = daily_agg.apply(
    lambda row: row['sentiment_x_magnitude_sum'] / row['magnitude_sum'] if row['magnitude_sum'] > 0 else 0,
    axis=1
)
daily_agg = daily_agg.drop(columns=['sentiment_x_magnitude_sum'])

# --- Step 6: Create Full Date Range (All Trading Days) ---
print(f"Creating full business day range from {START_DATE} to {END_DATE}...")
date_range = pd.bdate_range(start=START_DATE, end=END_DATE)
daily_complete = pd.DataFrame({'date': date_range})

# --- Step 7: Merge and Fill Missing Values ---
print("Merging aggregated data with full date range and filling NaNs...")
daily_df = daily_complete.merge(daily_agg, on='date', how='left')

fill_values = {
    'sentiment_mean': 0.0,
    'sentiment_std': 0.0,
    'sentiment_max': 0.0,
    'sentiment_min': 0.0,
    'news_count': 0,
    'magnitude_mean': 0.0,
    'magnitude_sum': 0.0,
    'sentiment_weighted_mean': 0.0
}
daily_df = daily_df.fillna(value=fill_values)
daily_df['news_count'] = daily_df['news_count'].astype(int)

# --- Step 8: Add Rolling Window Features ---
print(f"Creating rolling window features for windows: {ROLLING_WINDOWS}...")
for w in ROLLING_WINDOWS:
    daily_df[f'sentiment_mean_roll_avg_{w}d'] = daily_df['sentiment_mean'].rolling(window=w).mean()
    daily_df[f'sentiment_weighted_mean_roll_avg_{w}d'] = daily_df['sentiment_weighted_mean'].rolling(window=w).mean()
    daily_df[f'news_count_roll_sum_{w}d'] = daily_df['news_count'].rolling(window=w).sum()
    daily_df[f'sentiment_std_roll_max_{w}d'] = daily_df['sentiment_std'].rolling(window=w).max()

daily_df = daily_df.fillna(0.0)

# --- Step 9: Save Final Dataset ---
print(f"Saving final model-ready dataset to {OUTPUT_FILE}...")
daily_df.to_csv(OUTPUT_FILE, index=False, float_format='%.6f')

print("\n--- Script Finished ---")
print(f"Final DataFrame shape: {daily_df.shape}")
print(f"Output saved to: {OUTPUT_FILE}")

print("\nFinal DataFrame Info:")
daily_df.info()

print("\nSample of rows with news data (if any):")
news_sample = daily_df[daily_df['news_count'] > 0]
if not news_sample.empty:
    print(news_sample.head())
else:
    print("No news data was found in the processed files.")