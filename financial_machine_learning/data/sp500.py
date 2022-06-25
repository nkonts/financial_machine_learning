"""
Download the S&P 500 consitutiens from yfinance.
"""
import yfinance as yf
import bs4 as bs
import requests

def current_sp500_tickers():
    """Scrapes the current S&P 500 tickers from wikipedia.

    Returns:
        list[str]: List of tickers
    """
    resp = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, "html")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text
        if not "." in ticker:
            tickers.append(ticker.replace("\n",""))
    return tickers

def current_sp500_data(start_date: str = "2020-01-01", col: str = "Adj Close"):
    """Downloads the S&P 500 constituents from yfinance.

    Args:
        start_date (str, optional): The starting date of daily data. Defaults to "2020-01-01".
        col (str, optional): The column to select from yfinance (e.g. Close, Volume).
                             Defaults to "Adj Close".

    Returns:
        pd.DataFrame: A dataframe with tickers as columns
    """
    tickers = current_sp500_tickers()
    prices = yf.download(tickers, start=start_date)[col]
    return prices
