import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon (only needed once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_headline(headline):
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    scores = sid.polarity_scores(headline)
    
    # Determine sentiment category
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'headline': headline,
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'sentiment': sentiment
    }

def analyze_headlines(headlines):
    results = []
    for headline in headlines:
        results.append(analyze_headline(headline))
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    print("News Headline Sentiment Analyzer using VADER")
    print("Enter 'q' to quit")
    
    headlines = []
    while True:
        headline = input("\nEnter a news headline: ")
        if headline.lower() == 'q':
            break
        
        headlines.append(headline)
    
    if headlines:
        results = analyze_headlines(headlines)
        print("\nAnalysis Results:")
        print(results[['headline', 'sentiment', 'compound']])