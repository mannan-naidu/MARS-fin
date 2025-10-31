# ===================================================
# Twitter Scraper for Adani & Asian Paints
# Only for Stock Market Open Days (2015–2025)
# Saves separate CSVs for each company
# ===================================================

import snscrape.modules.twitter as sntwitter
import pandas as pd
import yfinance as yf
from datetime import timedelta
import time
import os

# -------------------------------
# STEP 1: Get stock market open days (NSE)
# -------------------------------
print("Fetching NSE trading days...")
data = yf.download("^NSEI", start="2015-01-01", end="2025-10-30")
market_days = data.index.date  # extract trading dates
print(f"✅ Total trading days found: {len(market_days)}")

# -------------------------------
# STEP 2: Scrape tweets
# -------------------------------
keywords = ["Adani", "Asian Paints"]

# Create directory to store results
os.makedirs("tweet_data", exist_ok=True)

print("\nStarting tweet scraping...\n")

for kw in keywords:
    company_tweets = []
    print(f"=== Scraping tweets for '{kw}' ===")

    for day in market_days:
        next_day = day + timedelta(days=1)
        query = f"{kw} since:{day} until:{next_day} lang:en"
        print(f"🔹 {kw} | {day}")

        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= 100:  # limit per day per keyword
                    break
                company_tweets.append([day, tweet.content])
        except Exception as e:
            print(f"⚠️ Error on {day} for {kw}: {e}")
            continue

        time.sleep(1)  # prevent rate limits

    # Convert to DataFrame
    df = pd.DataFrame(company_tweets, columns=["Date", "Tweet"])
    filename = f"tweet_data/{kw.lower().replace(' ', '_')}_tweets.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"💾 Saved {len(df)} tweets to '{filename}'\n")

print("\n✅ All tweets scraped and saved successfully!")
