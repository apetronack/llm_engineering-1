import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import quote
import time

def get_company_news(ticker):
    """
    Fetches recent news articles about a company from major business news sources.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    news_sources = {
        # Using MarketWatch's ticker-specific news page instead of search
        'MarketWatch': f'https://www.marketwatch.com/investing/stock/{ticker.lower()}/news',
        
        # Using Yahoo Finance's news feed (more reliable than Bloomberg/Reuters)
        'Yahoo Finance': f'https://finance.yahoo.com/quote/{ticker}/news/',
        
        # Using Seeking Alpha's news feed
        'Seeking Alpha': f'https://seekingalpha.com/symbol/{ticker}/news',
        
        # Using Benzinga's stock-specific news
        'Benzinga': f'https://www.benzinga.com/stock/{ticker.lower()}',

        # Using Bloomberg's search feature
        'Bloomberg': f'https://www.bloomberg.com/search?query={ticker}',

        # Using Reuters' search feature
        'Reuters': f'https://www.reuters.com/site-search/?query={ticker}'
    }
    
    results = {}
    
    for source, url in news_sources.items():
        try:
            print(f"Fetching news from {source}...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            if source == 'MarketWatch':
                # Try multiple possible selectors
                article_elements = (
                    soup.select('.article__content a') or  # Primary selector
                    soup.select('.elements a') or          # Alternative selector
                    soup.select('a[href*="/articles/"]')   # Backup selector
                )[:2]
                
                for element in article_elements:
                    href = element.get('href', '')
                    if href and '/articles/' in href:
                        if not href.startswith('http'):
                            href = 'https://www.marketwatch.com' + href
                        articles.append(href)
            
            elif source == 'Yahoo Finance':
                article_elements = soup.select('h3 a[href*="/news/"]')[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if href:
                        if not href.startswith('http'):
                            href = 'https://finance.yahoo.com' + href
                        articles.append(href)
            
            elif source == 'Seeking Alpha':
                article_elements = soup.select('a[data-test-id="post-list-item-title"]')[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if href:
                        if not href.startswith('http'):
                            href = 'https://seekingalpha.com' + href
                        articles.append(href)
            
            elif source == 'Benzinga':
                article_elements = (
                    soup.select('.content-listing-link') or
                    soup.select('a[href*="/news/"]')
                )[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if href and '/news/' in href:
                        if not href.startswith('http'):
                            href = 'https://www.benzinga.com' + href
                        articles.append(href)
            
            # Print debugging information
            print(f"Found {len(articles)} articles for {source}")
            if not articles:
                print(f"No articles found for {source}. Response status: {response.status_code}")
                print("HTML snippet:")
                print(soup.prettify()[:500])  # Print first 500 chars of HTML for debugging
            
            results[source] = articles
            time.sleep(2)  # Increased delay between requests
            
        except Exception as e:
            print(f"Error fetching {source} articles: {str(e)}")
            results[source] = [f"Error fetching {source} articles: {str(e)}"]
    
    return results

# Helper function to test the scraper
def test_news_scraper(ticker):
    """
    Test function to check news scraping for a given ticker
    """
    print(f"\nTesting news scraping for {ticker}")
    results = get_company_news(ticker)
    
    for source, articles in results.items():
        print(f"\n{source}:")
        if isinstance(articles, list):
            if articles:
                for i, article in enumerate(articles, 1):
                    print(f"{i}. {article}")
            else:
                print("No articles found")
        else:
            print(articles)  # Print error message

if __name__ == "__main__":
    # Test the scraper
    test_news_scraper('AAPL')