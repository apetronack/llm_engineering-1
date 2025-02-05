import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import quote
import time
import json
from newspaper import Article
import ollama

LLAMA_MODEL = "llama3.2"

def get_stock_info(ticker_symbol):
    """Previous get_stock_info function remains the same"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        history = ticker.history(start=start_date, end=end_date)
        
        stock_data = {
            'symbol': ticker_symbol,
            'company_name': info.get('longName', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
            'previous_close': info.get('previousClose', 'N/A'),
            'open': info.get('open', 'N/A'),
            'day_high': info.get('dayHigh', 'N/A'),
            'day_low': info.get('dayLow', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'fifty_day_avg': info.get('fiftyDayAverage', 'N/A'),
            'two_hundred_day_avg': info.get('twoHundredDayAverage', 'N/A'),
            'year_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'year_low': info.get('fiftyTwoWeekLow', 'N/A')
        }
        
        return stock_data
        
    except Exception as e:
        return f"Error fetching stock data for {ticker_symbol}: {str(e)}"

def get_company_news(ticker):
    """Previous get_company_news function remains the same"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    news_sources = {
        'Bloomberg': f'https://www.bloomberg.com/search?query={quote(ticker)}',
        'Reuters': f'https://www.reuters.com/search/news?blob={quote(ticker)}',
        'Business Insider': f'https://www.businessinsider.com/s?q={quote(ticker)}',
        'MarketWatch': f'https://www.marketwatch.com/search?q={quote(ticker)}&m=Ticker'
    }
    
    results = {}
    
    for source, url in news_sources.items():
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            if source == 'Bloomberg':
                article_elements = soup.select('article h1 a')[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if not href.startswith('http'):
                        href = 'https://www.bloomberg.com' + href
                    articles.append(href)
                    
            elif source == 'Reuters':
                article_elements = soup.select('.search-result-content a')[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if not href.startswith('http'):
                        href = 'https://www.reuters.com' + href
                    articles.append(href)
                    
            elif source == 'Business Insider':
                article_elements = soup.select('h2.tout-title a')[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if not href.startswith('http'):
                        href = 'https://www.businessinsider.com' + href
                    articles.append(href)
                    
            elif source == 'MarketWatch':
                article_elements = soup.select('.article__headline a')[:2]
                for element in article_elements:
                    href = element.get('href', '')
                    if not href.startswith('http'):
                        href = 'https://www.marketwatch.com' + href
                    articles.append(href)
            
            results[source] = articles
            time.sleep(1)
            
        except Exception as e:
            results[source] = [f"Error fetching {source} articles: {str(e)}"]
    
    return results

def extract_article_content(url):
    """
    Extracts the title and content from a news article URL using newspaper3k.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        return {
            'title': article.title,
            'text': article.text
        }
    except Exception as e:
        return {
            'title': 'Error extracting article',
            'text': f'Failed to extract article content: {str(e)}'
        }

def analyze_article_sentiment(article_content, company_name):
    """
    Uses Llama to analyze article content and create a brief summary with sentiment.
    """
    prompt = f"""
    Please analyze the following article about {company_name} and provide a summary in 5 sentences or less.
    Focus on the company outlook (optimistic or pessimistic) and any specific factors mentioned.
    
    Article Title: {article_content['title']}
    
    Article Content: {article_content['text']}
    
    Please structure your response as:
    Summary: [Your 5-sentence summary]
    Sentiment: [Optimistic/Pessimistic/Neutral]
    Key Factors: [List main factors influencing the outlook]
    """
    
    try:
        response = ollama.chat(model=LLAMA_MODEL, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error analyzing article: {str(e)}"

def generate_company_outlook(stock_data, article_analyses):
    """
    Uses Llama to generate a near-term outlook based on stock data and article analyses.
    """
    prompt = f"""
    Please analyze the following information about {stock_data['company_name']} ({stock_data['symbol']}) 
    and provide a near-term outlook. This is for research purposes only, not financial advice.
    
    Stock Information:
    Current Price: ${stock_data['current_price']}
    Previous Close: ${stock_data['previous_close']}
    50-Day Average: ${stock_data['fifty_day_avg']}
    200-Day Average: ${stock_data['two_hundred_day_avg']}
    
    Recent News Analyses:
    {article_analyses}
    
    Please provide:
    1. A 2-3 sentence summary of the current situation
    2. 2-3 Key factors influencing the company
    3. 2-3 Potential risks and opportunities
    4. 1 sentence near-term outlook broken into next 1-2 months, 3-6 months, and 6-12 months. 
    5. 1-3 sentence analysis of the company fundamentals and market conditions
    6. Recommendation: Buy/Sell/Hold

    Remember: This is speculative analysis for research purposes only.
    """
    
    try:
        response = ollama.chat(model=LLAMA_MODEL, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error generating outlook: {str(e)}"

def format_stock_info(stock_data):
    """Previous format_stock_info function remains the same"""
    if isinstance(stock_data, str):
        return stock_data
        
    return f"""
Stock Information for {stock_data['company_name']} ({stock_data['symbol']}):
----------------------------------------
Current Price: ${stock_data['current_price']}
Previous Close: ${stock_data['previous_close']}
Today's Range: ${stock_data['day_low']} - ${stock_data['day_high']}
Volume: {stock_data['volume']:,}
Market Cap: ${stock_data['market_cap']:,}
P/E Ratio: {stock_data['pe_ratio']}

Moving Averages:
50-Day: ${stock_data['fifty_day_avg']}
200-Day: ${stock_data['two_hundred_day_avg']}

52-Week Range:
Low: ${stock_data['year_low']}
High: ${stock_data['year_high']}
"""

def get_company_analysis():
    """
    Enhanced interactive function that includes AI analysis of articles and company outlook.
    """
    while True:
        ticker = input("\nEnter company ticker symbol (or 'quit' to exit): ").upper()
        
        if ticker.lower() == 'quit':
            print("Goodbye!")
            break
            
        print(f"\nFetching information for {ticker}...")
        
        # Get stock information
        stock_data = get_stock_info(ticker)
        print(format_stock_info(stock_data))
        
        # Get news articles
        print("\nFetching and analyzing latest news articles...")
        news_data = get_company_news(ticker)
        
        # Analyze articles
        article_analyses = []
        for source, urls in news_data.items():
            print(f"\nAnalyzing articles from {source}...")
            for url in urls:
                article_content = extract_article_content(url)
                analysis = analyze_article_sentiment(article_content, stock_data['company_name'])
                article_analyses.append(f"Source: {source}\nURL: {url}\n{analysis}\n")
                print(f"Analyzed: {article_content['title']}")
        
        # Generate company outlook
        print("\nGenerating company outlook...")
        outlook = generate_company_outlook(stock_data, "\n".join(article_analyses))
        
        # Print full analysis
        print("\nArticle Analyses:")
        print("----------------------------------------")
        for analysis in article_analyses:
            print(analysis)
            print("----------------------------------------")
        
        print("\nCompany Outlook:")
        print("----------------------------------------")
        print(outlook)
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("Welcome to the AI-Enhanced Stock and News Analyzer!")
    print("This tool provides real-time stock information, news analysis, and AI-generated insights.")
    print("Note: This is for research purposes only, not financial advice.")
    get_company_analysis()