import pandas as pd
import numpy as np
import traceback
import sys
import os

print("Starting the dataset merging process for Asian Paints (v8 fixing base date format)...")

# --- Configuration ---
OUTPUT_FILE = 'model_ready_merged_dataset_asianpaints.csv'
START_DATE_FINAL = '2015-01-01' # User specified start date

# --- FILENAMES TO CHECK ---
BASE_FILE = 'ASIANPAINT.csv'
SENTIMENT_FILE = 'daily_news_features_finbert_asianpaints.csv'
MACRO_FILE = 'macro_data_fred_detailed filled.csv'
QUARTERLY_FILE = 'asianpaints quarterly data.csv'
ANNOUNCEMENT_DATE_FILE = 'asianpaints quarterly release dates.csv'

FILES_TO_LOAD = [BASE_FILE, SENTIMENT_FILE, MACRO_FILE, QUARTERLY_FILE, ANNOUNCEMENT_DATE_FILE]

# --- Check for files first ---
missing_files = [f for f in FILES_TO_LOAD if not os.path.exists(f)]
if missing_files:
    print(f"--- ERROR ---")
    print(f"Cannot find the following file(s): {', '.join(missing_files)}")
    print("Please make sure all 5 files are in the same directory as the script.")
    sys.exit()

try:
    # --- 1. Load Announcement Dates from CSV ---
    print(f"Loading Quarterly Announcement Dates ({ANNOUNCEMENT_DATE_FILE})...")
    try:
        df_announce_dates = pd.read_csv(ANNOUNCEMENT_DATE_FILE, encoding='utf-8-sig', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading announcement date CSV '{ANNOUNCEMENT_DATE_FILE}': {e}")
        sys.exit()

    if 'Fiscal Year' not in df_announce_dates.columns:
        print(f"--- ERROR --- 'Fiscal Year' column not found in {ANNOUNCEMENT_DATE_FILE}.")
        sys.exit()
    quarter_cols = [col for col in df_announce_dates.columns if col not in ['Fiscal Year']]
    if not quarter_cols:
         print(f"--- ERROR --- Could not find quarter columns in {ANNOUNCEMENT_DATE_FILE}.")
         sys.exit()
    df_announce_dates = df_announce_dates.melt(id_vars=['Fiscal Year'], value_vars=quarter_cols,
                                             var_name='Quarter', value_name='Announcement Date')
    def map_quarter_to_date(row):
        # ... [Mapping logic remains the same] ...
        fiscal_year_str = str(row['Fiscal Year'])
        try:
            year_part = fiscal_year_str.split(' ')[-1]
            start_year = int(year_part.split('-')[0])
        except (IndexError, ValueError):
             try: start_year = int(fiscal_year_str)
             except (ValueError, TypeError): return None
        quarter_str = str(row['Quarter'])
        if 'Q1' in quarter_str or 'Apr-Jun' in quarter_str: return pd.Timestamp(year=start_year, month=6, day=30)
        elif 'Q2' in quarter_str or 'Jul-Sep' in quarter_str: return pd.Timestamp(year=start_year, month=9, day=30)
        elif 'Q3' in quarter_str or 'Oct-Dec' in quarter_str: return pd.Timestamp(year=start_year, month=12, day=31)
        elif 'Q4' in quarter_str or 'Jan-Mar' in quarter_str: return pd.Timestamp(year=start_year + 1, month=3, day=31)
        else: return None
    df_announce_dates['Quarter End Date'] = df_announce_dates.apply(map_quarter_to_date, axis=1)
    df_announce_dates['Announcement Date'] = pd.to_datetime(df_announce_dates['Announcement Date'], errors='coerce')
    df_announce_dates = df_announce_dates.dropna(subset=['Quarter End Date', 'Announcement Date'])
    df_announce_map = df_announce_dates[['Quarter End Date', 'Announcement Date']].set_index('Quarter End Date')
    print(f"Announcement date mapping created. {len(df_announce_map)} entries found.")
    if df_announce_map.empty:
        print("--- ERROR --- No valid announcement date mappings found.")
        sys.exit()

    # --- 2. Load and Transform Quarterly Data from CSV ---
    print(f"Loading and Transforming Quarterly Data ({QUARTERLY_FILE})...")
    try:
        df_q = pd.read_csv(QUARTERLY_FILE, index_col=0, encoding='latin1')
    except Exception as e:
        print(f"--- ERROR reading quarterly data CSV '{QUARTERLY_FILE}'. Error: {e}")
        sys.exit()
    df_q = df_q.T
    try:
        # ... [Date parsing logic remains the same] ...
        parsed_dates = None
        for fmt in ["%b '%y", "%b-%y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%b %Y"]:
             try:
                 temp_dates = pd.to_datetime(df_q.index, format=fmt, errors='coerce')
                 if not temp_dates.isna().all():
                     parsed_dates = temp_dates + pd.offsets.MonthEnd(0)
                     print(f"Successfully parsed quarterly dates with format: {fmt}")
                     break
             except (ValueError, TypeError): continue
        if parsed_dates is None or parsed_dates.isna().any():
             print("Specific formats failed, attempting automatic date inference...")
             parsed_dates = pd.to_datetime(df_q.index, errors='coerce') + pd.offsets.MonthEnd(0)
             if parsed_dates.isna().any(): raise ValueError("Could not parse quarterly index dates.")
             else: print("Successfully parsed quarterly dates automatically.")
        df_q.index = parsed_dates
    except ValueError as e:
        print(f"--- ERROR parsing quarterly report dates: {e}")
        sys.exit()
    df_q.index.name = 'Quarter End Date'
    df_q = df_q.sort_index()
    df_q.columns = df_q.columns.str.replace('[ /]', '_', regex=True).str.replace('[^A-Za-z0-9_]+', '', regex=True)
    df_q = df_q.replace({',': '', '"': '', '%': ''}, regex=True).apply(pd.to_numeric, errors='coerce')
    df_q = df_q.dropna(axis=1, how='all')
    print(f"Quarterly data loaded. Shape: {df_q.shape}")

    # --- 3. Merge Quarterly Data with Announcement Dates & Set Index ---
    print("Mapping announcement dates to quarterly data...")
    df_q_merged = df_q.join(df_announce_map, how='inner')
    if 'Announcement Date' not in df_q_merged.columns or df_q_merged['Announcement Date'].isnull().all():
        print("--- ERROR --- Could not map Announcement Dates to Quarterly Data.")
        sys.exit()
    df_q_final = df_q_merged.set_index('Announcement Date')
    df_q_final = df_q_final.sort_index()
    print(f"Quarterly data index set to Announcement Date. Shape: {df_q_final.shape}")

    # --- 4. Load Base Financial Data ---
    print(f"Loading Base Financial Data ({BASE_FILE})...")
    try:
        df_base = pd.read_csv(BASE_FILE, header=0, skiprows=[1])
        df_base = df_base.dropna(how='all')
        if 'Date' in df_base.columns: date_col = 'Date'
        elif 'date' in df_base.columns: date_col = 'date'
        else:
             date_col = df_base.columns[0]
             print(f"Warning: Using first column '{date_col}' as date in {BASE_FILE}.")

        # --- THIS IS THE FIX ---
        # Remove dayfirst=True as the format seems to be YYYY-MM-DD
        df_base['date'] = pd.to_datetime(df_base[date_col], errors='coerce')
        # --- END OF FIX ---

        df_base = df_base.dropna(subset=['date'])
        df_base = df_base.set_index('date')
        df_base = df_base.sort_index()
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df_base.columns for col in ohlcv_cols):
             print(f"Warning: Standard OHLCV columns not found in {BASE_FILE}.")
             numeric_cols = df_base.select_dtypes(include=np.number).columns.tolist()
             if len(numeric_cols) >= 5: ohlcv_cols = numeric_cols[:5]
             else: raise ValueError("Could not find sufficient numeric columns for OHLCV.")
        df_base = df_base[ohlcv_cols]
        for col in df_base.columns: df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
        print(f"Base data loaded. Shape: {df_base.shape}")
    except Exception as e:
         print(f"--- ERROR loading or processing base file {BASE_FILE}: {e}")
         traceback.print_exc()
         sys.exit()

    # --- 5. Load Sentiment Data ---
    print(f"Loading Sentiment Data ({SENTIMENT_FILE})...")
    df_news = pd.read_csv(SENTIMENT_FILE)
    df_news['date'] = pd.to_datetime(df_news['date']) # Assume YYYY-MM-DD
    df_news = df_news.set_index('date')
    df_news = df_news.sort_index()
    print(f"Sentiment data loaded. Shape: {df_news.shape}")

    # --- 6. Load Macroeconomic Data ---
    print(f"Loading Macroeconomic Data ({MACRO_FILE})...")
    df_macro = pd.read_csv(MACRO_FILE)
    df_macro['date'] = pd.to_datetime(df_macro['date'], dayfirst=True) # Keep dayfirst=True here
    df_macro = df_macro.set_index('date')
    df_macro = df_macro.sort_index()
    print(f"Macro data loaded. Shape: {df_macro.shape}")

    # --- 7. Filter Base Data ---
    print(f"Filtering base data (backbone) from {START_DATE_FINAL} onwards.")
    df_base_filtered = df_base.loc[START_DATE_FINAL:]
    print(f"Base data filtered. New shape: {df_base_filtered.shape}")

    # --- 8. Merge Datasets ---
    print("Joining Base Financials + Sentiment Data...")
    df_final = df_base_filtered.join(df_news, how='left')
    print("Joining + Macro Data...")
    df_final = df_final.join(df_macro, how='left')
    print("Reindexing and Forward-Filling Quarterly Data (using Announcement Dates)...")
    df_q_final.index = df_q_final.index.tz_localize(None)
    df_final.index = df_final.index.tz_localize(None)
    df_q_filled = df_q_final.reindex(df_final.index, method='ffill')
    print("Joining + Forward-Filled Quarterly Data...")
    df_final = df_final.join(df_q_filled, rsuffix='_quarterly')

    # --- 9. Final Cleaning (Handling NaNs) ---
    print("Cleaning final dataset (filling NaNs)...")
    sentiment_cols = [col for col in df_final.columns if 'sentiment_' in col or 'news_count' in col]
    if sentiment_cols:
        df_final[sentiment_cols] = df_final[sentiment_cols].fillna(0)
        print(f"Filled NaNs with 0 for {len(sentiment_cols)} sentiment columns.")

    other_cols = [col for col in df_final.columns if col not in sentiment_cols]
    df_final[other_cols] = df_final[other_cols].ffill()

    quarterly_cols_in_final = [col for col in df_final.columns if '_quarterly' in col]
    if not quarterly_cols_in_final: quarterly_cols_in_final = [col for col in df_q_final.columns if col in df_final.columns]
    if quarterly_cols_in_final:
         df_final[quarterly_cols_in_final] = df_final[quarterly_cols_in_final].bfill()
         print(f"Back-filled NaNs at the start for {len(quarterly_cols_in_final)} quarterly columns.")

    df_final[other_cols] = df_final[other_cols].bfill()
    print("Filled any remaining NaNs for other columns using ffill/bfill.")

    # --- 10. Save Final Dataset ---
    if df_final.index.name is None: df_final.index.name = 'date'
    df_final.to_csv(OUTPUT_FILE)
    print(f"\n--- Script Finished ---")
    print(f"Final merged dataset saved to: {OUTPUT_FILE}")
    print(f"Final DataFrame shape: {df_final.shape}")

    # --- 11. Verification ---
    # ... [Verification logic remains the same] ...
    print("\nVerifying Quarterly FFill (showing data around an announcement date):")
    try:
        announce_dates_in_range = df_q_final.index[(df_q_final.index >= df_final.index.min()) & (df_q_final.index <= df_final.index.max())]
        if not announce_dates_in_range.empty:
            target_announce_date = announce_dates_in_range[1] if len(announce_dates_in_range) > 1 else announce_dates_in_range[0]
            print(f"Checking fill around announcement date: {target_announce_date.strftime('%Y-%m-%d')}")
            q_col_name_base = next((col for col in df_q_final.columns if 'Sales' in col or 'Income' in col), df_q_final.columns[0])
            q_col_to_check = f"{q_col_name_base}_quarterly" if f"{q_col_name_base}_quarterly" in df_final.columns else q_col_name_base

            date_before = target_announce_date - pd.Timedelta(days=1)
            while date_before not in df_final.index and date_before >= df_final.index.min(): date_before -= pd.Timedelta(days=1)
            date_after = target_announce_date + pd.Timedelta(days=1)
            while date_after not in df_final.index and date_after <= df_final.index.max(): date_after += pd.Timedelta(days=1)

            if date_before in df_final.index and target_announce_date in df_final.index and date_after in df_final.index:
                value_before = df_final.loc[date_before, q_col_to_check]
                value_on = df_final.loc[target_announce_date, q_col_to_check]
                value_after = df_final.loc[date_after, q_col_to_check]
                print(f"Value day before '{date_before.strftime('%Y-%m-%d')}' ({q_col_to_check}): {value_before}")
                print(f"Value on announcement day '{target_announce_date.strftime('%Y-%m-%d')}' ({q_col_to_check}): {value_on}")
                print(f"Value day after '{date_after.strftime('%Y-%m-%d')}' ({q_col_to_check}): {value_after}")
                if abs(value_before - value_on) > 1e-6 and abs(value_on - value_after) < 1e-6 : print(">>> Verification Successful: Quarterly data appears to fill correctly from announcement date.")
                else: print(">>> Verification Warning: Quarterly data fill pattern looks unusual. Please check manually.")
                print("\nShowing data slice around the announcement date:")
                start_slice, end_slice = date_before, date_after
                display_cols = ['Close', 'sentiment_weighted_mean', q_col_to_check]
                display_cols = [col for col in display_cols if col in df_final.columns]
                print(df_final.loc[start_slice:end_slice][display_cols])
            else: print("Could not find suitable dates before/on/after announcement date in the final index.")
        else: print("No announcement dates found within the final dataset range for verification.")
    except Exception as e: print(f"Could not perform verification check. Error: {e}")

    print("\nFinal Data Head:")
    print(df_final.head())


except FileNotFoundError as e:
     print(f"--- ERROR: File Not Found ---")
     print(f"Could not find: {e.filename}")
     print("Please ensure all files are in the same directory.")
except Exception as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    traceback.print_exc()