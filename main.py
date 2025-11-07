import pandas as pd
import numpy as np

# Import your modules
from portfolio_optimizer.data import get_annualized_inputs
from portfolio_optimizer.model import get_portfolio_stats
from portfolio_optimizer.optimization import (
    find_max_sharpe_portfolio,
    find_min_volatility_portfolio,
    calculate_efficient_frontier
)
# NEW: Import the plotting function
from portfolio_optimizer.plots import plot_results

# Define your project parameters
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.02 # 2%

def run_analysis():
    print("Starting Portfolio Analysis...")
    print(f"Fetching data for: {', '.join(TICKERS)}\n")
    
    try:
        # 1. Get data
        mean_returns, cov_matrix = get_annualized_inputs(
            tickers=TICKERS,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # --- 2. Calculate Optimal Portfolios ---
        print("Calculating Optimal Portfolios...")
        max_sharpe_weights = find_max_sharpe_portfolio(
            mean_returns, cov_matrix, RISK_FREE_RATE
        )
        min_vol_weights = find_min_volatility_portfolio(
            mean_returns, cov_matrix
        )
        
        # Get stats for optimal portfolios
        max_sharpe_stats = get_portfolio_stats(
            max_sharpe_weights, mean_returns, cov_matrix
        )
        min_vol_stats = get_portfolio_stats(
            min_vol_weights, mean_returns, cov_matrix
        )
        
        # Store in a dictionary for the plot
        optimal_portfolios = {
            "Max Sharpe Ratio": max_sharpe_stats,
            "Min Volatility": min_vol_stats
        }

        # --- 3. Calculate Sample Portfolios (for Analysis) ---
        print("Calculating Sample Portfolios...")
        num_assets = len(TICKERS)
        
        # Sample 1: Equally Weighted (1/n)
        equal_weights = np.array([1 / num_assets] * num_assets)
        equal_stats = get_portfolio_stats(
            equal_weights, mean_returns, cov_matrix
        )
        
        # Sample 2 & 3 are the individual assets, but we'll
        # calculate them below. We can add more samples here
        # if we want, but the prompt's 3 samples are covered
        # by (1/n) and the individual asset plots.
        
        sample_portfolios = {
            "Equally Weighted (1/n)": equal_stats
        }
        
        # --- 4. Calculate Individual Asset Stats (for plotting) ---
        # (This also covers Sample 2 & 3 from the prompt)
        print("Calculating Individual Asset Stats...")
        individual_assets = {}
        for ticker in TICKERS:
            # Create a 100% weight for this one asset
            weights = np.zeros(num_assets)
            weights[TICKERS.index(ticker)] = 1.0
            
            # Get its stats
            asset_stats = get_portfolio_stats(weights, mean_returns, cov_matrix)
            individual_assets[ticker] = asset_stats
            
        # --- 5. Calculate the Efficient Frontier ---
        print("Calculating Efficient Frontier...")
        frontier_returns, frontier_volatilities = calculate_efficient_frontier(
            mean_returns, cov_matrix
        )
        frontier_data = (frontier_volatilities, frontier_returns)
        
        # --- 6. Plot All Results ---
        plot_results(
            frontier_data=frontier_data,
            optimal_portfolios=optimal_portfolios,
            sample_portfolios=sample_portfolios,
            individual_assets=individual_assets
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    pd.set_option('display.width', 100)
    pd.set_option('display.precision', 4)
    run_analysis()