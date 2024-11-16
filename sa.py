import streamlit as st
import yfinance as yf
from textblob import TextBlob
import requests

NEWS_API_KEY = ""

def fetch_news(stock_symbol):
    """Fetches news articles using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    articles = []
    if data["status"] == "ok":
        for article in data["articles"]:
            articles.append(article["url"])
    return articles[:20]  # Limit to top 20 articles

def analyze_sentiment(url):
    """Analyzes the sentiment of the article at the given URL."""
    response = requests.get(url)
    article_text = response.text
    
    analysis = TextBlob(article_text)
    return analysis.sentiment.polarity

def interpret_score(avg_sentiment):
    """Interprets the average sentiment score."""
    if avg_sentiment > 0.5:
        return "Highly Positive: The overall market sentiment is very positive, indicating strong bullish trends."
    elif 0.1 < avg_sentiment <= 0.5:
        return "Positive: The sentiment is generally positive, suggesting mild bullish trends."
    elif -0.1 <= avg_sentiment <= 0.1:
        return "Neutral: The sentiment is mixed or neutral, with no strong direction in market sentiment."
    elif -0.5 <= avg_sentiment < -0.1:
        return "Negative: The sentiment is generally negative, indicating mild bearish trends."
    else:
        return "Highly Negative: The overall market sentiment is very negative, suggesting strong bearish trends."

def main():
    st.title("Market Sentiment Analyzer")
    
    # Input: Stock Symbol
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    if st.button("Analyze Sentiment"):
        st.write(f"Fetching news for {stock_symbol}...")
        
        # Fetch news articles
        news_urls = fetch_news(stock_symbol)
        
        if news_urls:
            st.write(f"Found {len(news_urls)} articles. Analyzing sentiment...")
            
            sentiments = []
            for url in news_urls:
                st.write(f"Analyzing: {url}")
                sentiment = analyze_sentiment(url)
                sentiments.append(sentiment)
                st.write(f"Sentiment Score: {sentiment}")
            
            # Calculate the average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            st.write(f"Average Sentiment Score: {avg_sentiment}")
            
            # Interpret the score
            interpretation = interpret_score(avg_sentiment)
            st.write(interpretation)
        else:
            st.error("No news articles found for the given stock symbol.")

if __name__ == "__main__":
    main()
