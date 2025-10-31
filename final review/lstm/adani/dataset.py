import pandas as pd
import numpy as np

# --- Configuration ---
MASTER_FILE = 'ADANIENT_master_data_engineered.csv'
NEWS_FILE = 'adani news.csv' # Your file with raw headlines
OUTPUT_FILE = 'ADANIENT_master_data_FINAL.csv' # The new, final dataset

# --- 1. Define Keyword Categories for Multi-Hot Encoding ---
CATEGORIES = {
    'is_fraud_allegation': [
        'hindenburg', 'indictment', 'bribery', 'fraud', 'charges', 
        'allegation', 'misconduct', 'bombshell'
    ],
    'is_regulatory_probe': [
        'sebi', 'probe', 'investigation', 'pac', 'regulatory', 
        'public accounts committee', 'justice'
    ],
    'is_financial_positive': [
        'q3 fy24 results', 'q4 fy24 profit', 'fy24 full year results', 
        'record 420 mmt', 'recovers past', 'erases all hindenburg-induced losses',
        'operational performance'
    ],
    'is_financial_negative': [
        'stock falls', 'stock plunge', 'market cap loss', 'crash', 
        'cancels $600 million', 'cancels $2 billion'
    ],
    'is_company_response': [
        'denies charges', 'breaks silence', 'baseless', 'reiterates',
        'dismisses allegations'
    ]
}

print("Starting Multi-Hot Encoding Feature Engineering...")

# --- 2. Load Datasets ---
try:
    df_master = pd.read_csv(MASTER_FILE)
    df_news = pd.read_csv(NEWS_FILE)
    print(f"Loaded {MASTER_FILE} (Shape: {df_master.shape})")
    print(f"Loaded {NEWS_FILE} (Shape: {df_news.shape})")
except Exception as e:
    print(f"--- ERROR loading files ---: {e}")
    exit()

# --- 3. Prepare DataFrames ---
print("Parsing master file dates (YYYY-MM-DD)...")
df_master['date'] = pd.to_datetime(df_master['date'])

print("Parsing news file dates (DD-MM-YYYY)...")
df_news['Date'] = pd.to_datetime(df_news['Date'], format='%d-%m-%Y') 
df_news.columns = ['date', 'headline']
df_news['headline_lower'] = df_news['headline'].str.lower()

# --- 4. Perform Multi-Hot Encoding ---
print("Creating multi-hot encoded category features...")
daily_categories = pd.DataFrame(index=df_news['date'].unique())
for cat_name in CATEGORIES.keys():
    daily_categories[cat_name] = 0

for cat_name, keywords in CATEGORIES.items():
    print(f"Processing category: {cat_name}...")
    keyword_pattern = '|'.join(keywords)
    matching_headlines = df_news[df_news['headline_lower'].str.contains(keyword_pattern, na=False)]
    matching_dates = matching_headlines['date'].unique()
    daily_categories.loc[matching_dates, cat_name] = 1

# Group by date (which is the index) and sum/clip
df_multi_hot = daily_categories.groupby(daily_categories.index).sum().clip(upper=1)

print("\nMulti-Hot Encoding Summary (first 5 days with news):")
print(df_multi_hot.query('index >= "2024-01-01"').head())

# --- 5. Merge with Master DataFrame ---
print(f"\nMerging new category features into {MASTER_FILE}...")

# ### <<< MODIFICATION >>> ###
# We merge df_master's 'date' COLUMN (left_on='date')
# with df_multi_hot's INDEX (right_index=True).
df_final = pd.merge(
    df_master, 
    df_multi_hot, 
    left_on='date', 
    right_index=True, 
    how='left'
)
# ### <<< END MODIFICATION >>> ###

# Fill any days with NO news with 0 for the category columns
category_columns = list(CATEGORIES.keys())
df_final[category_columns] = df_final[category_columns].fillna(0).astype(int)

# --- 6. Save the New, Final Dataset ---
try:
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n*** SUCCESS! ***")
    print(f"New dataset with {len(category_columns)} multi-hot features saved to:")
    print(f"{OUTPUT_FILE} (Shape: {df_final.shape})")
    
    print("\nNew columns added:")
    print(category_columns)
    
    print(f"\n--- Next Step ---")
    print(f"Now, please run the *next* script I provide, which will train the model on this new file.")

except Exception as e:
    print(f"--- ERROR saving final file ---: {e}")