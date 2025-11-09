import pandas as pd
import numpy as np
import yfinance as yf
from pandas import DataFrame

def fetch_price_data(tickers: list[str], start_date: str, end_date: str) -> DataFrame:
    """
    Fetches historical 'Adj Close' prices for a list of tickers.
    """
    # yfinance returns a DataFrame with a MultiIndex
    # We select 'Adj Close' and drop the top-level index
    prices = yf.download(tickers, start=start_date, end=end_date)
    
    if prices.empty:
        raise ValueError("No data fetched. Check tickers and date range.")
        
    prices = prices.get('Close')
    
    # If only one ticker, yf returns a Series, convert to DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        
    prices = prices.dropna() # Remove any rows with missing data
    return prices


def calculate_returns(prices: DataFrame) -> DataFrame:
    """
    Calculates daily logarithmic returns from a DataFrame of prices.
    """
    # We use log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Drop the first row of NaN values that results from shifting
    return log_returns.dropna()


def get_annualized_inputs(tickers: list[str], start_date: str, end_date: str,
                          trading_days: int = 252) -> tuple[DataFrame, DataFrame]:
    """
    Orchestrator function to fetch data, calculate returns,
    and return annualized mean returns and the covariance matrix.
    """
    # 1. Fetch price data
    prices = fetch_price_data(tickers, start_date, end_date)
    
    # 2. Calculate daily log returns
    log_returns = calculate_returns(prices)
    
    if log_returns.empty:
        raise ValueError("Return calculation resulted in empty DataFrame.")

    # 3. Calculate annualized mean returns
    # We multiply mean daily returns by the number of trading days
    mean_returns = log_returns.mean() * trading_days
    
    # 4. Calculate annualized covariance matrix
    # We multiply the daily covariance matrix by the number of trading days
    cov_matrix = log_returns.cov() * trading_days
    
    return mean_returns, cov_matrix