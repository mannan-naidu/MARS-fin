import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
lag_llama_df = pd.read_csv('lag_llama metrics.csv')
lstm_df = pd.read_csv('lstm_model_metrics.csv')

# Standardize column names for merging
lag_llama_df.rename(columns={'ticker': 'Stock_Symbol'}, inplace=True)

# Merge the two dataframes
merged_df = pd.merge(lag_llama_df, lstm_df, on='Stock_Symbol', suffixes=('_lag_llama', '_lstm'))

# Define sectors for each stock symbol
sector_mapping = {
    'ADANIENT': 'Conglomerate',
    'ADANIPORTS': 'Infrastructure',
    'APOLLOHOSP': 'Healthcare',
    'ASIANPAINT': 'Chemicals',
    'AXISBANK': 'Banking',
    'BAJAJ-AUTO': 'Automobile',
    'BAJAJFINSV': 'Financial Services',
    'BAJFINANCE': 'Financial Services',
    'BHARTIARTL': 'Telecommunication',
    'BPCL': 'Energy',
    'BRITANNIA': 'FMCG',
    'CIPLA': 'Pharmaceuticals',
    'COALINDIA': 'Energy',
    'DIVISLAB': 'Pharmaceuticals',
    'DRREDDY': 'Pharmaceuticals',
    'EICHERMOT': 'Automobile',
    'GRASIM': 'Materials',
    'HCLTECH': 'IT',
    'HDFCBANK': 'Banking',
    'HDFCLIFE': 'Financial Services',
    'HEROMOTOCO': 'Automobile',
    'HINDALCO': 'Materials',
    'HINDUNILVR': 'FMCG',
    'ICICIBANK': 'Banking',
    'INDUSINDBK': 'Banking',
    'INFY': 'IT',
    'ITC': 'FMCG',
    'JSWSTEEL': 'Materials',
    'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure',
    'M&M': 'Automobile',
    'MARUTI': 'Automobile',
    'NESTLEIND': 'FMCG',
    'NTPC': 'Energy',
    'ONGC': 'Energy',
    'POWERGRID': 'Energy',
    'RELIANCE': 'Conglomerate',
    'SBILIFE': 'Financial Services',
    'SBIN': 'Banking',
    'SUNPHARMA': 'Pharmaceuticals',
    'TATACONSUM': 'FMCG',
    'TATAMOTORS': 'Automobile',
    'TATASTEEL': 'Materials',
    'TCS': 'IT',
    'TECHM': 'IT',
    'TITAN': 'Consumer Goods',
    'ULTRACEMCO': 'Materials',
    'WIPRO': 'IT'
}
merged_df['Sector'] = merged_df['Stock_Symbol'].map(sector_mapping)

# Determine the winner for each stock based on lower MSE
merged_df['Winner'] = merged_df.apply(lambda row: 'Lag_LLaMA' if row['mse'] < row['MSE'] else 'LSTM', axis=1)

# Calculate win rates for each model in each sector
sector_win_rates = merged_df.groupby('Sector')['Winner'].value_counts(normalize=True).unstack().fillna(0)

# Create the visualization
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8))

sector_win_rates.plot(kind='bar', ax=ax, colormap='viridis', width=0.8)

# Add titles and labels
ax.set_title('Model Performance by Sector', fontsize=20, color='white')
ax.set_xlabel('Sector', fontsize=14, color='white')
ax.set_ylabel('Win Rate', fontsize=14, color='white')
ax.tick_params(axis='x', labelrotation=45, colors='white')
ax.tick_params(axis='y', colors='white')
ax.legend(title='Model', prop={'size': 12})
plt.tight_layout()

# Save the plot
plt.savefig('model_performance_by_sector.png')