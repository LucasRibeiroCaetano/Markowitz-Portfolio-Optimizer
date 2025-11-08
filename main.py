import pandas as pd
import numpy as np

# Estes imports respeitam a sua estrutura:
# data.py, model.py, optimization.py, plots.py
from portfolio_optimizer.data import get_annualized_inputs
from portfolio_optimizer.model import get_portfolio_stats
from portfolio_optimizer.optimization import (
    find_max_sharpe_portfolio,
    find_min_volatility_portfolio,
    calculate_efficient_frontier
)
from portfolio_optimizer.plots import plot_results

# --- UPDATED 10-ASSET LIST ---
TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 
    'NVDA', 'META', 'JPM', 'SPY', 'QQQ'
]
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
        
        max_sharpe_stats = get_portfolio_stats(
            max_sharpe_weights, mean_returns, cov_matrix
        )
        min_vol_stats = get_portfolio_stats(
            min_vol_weights, mean_returns, cov_matrix
        )
        
        optimal_portfolios = {
            "Max Sharpe Ratio": max_sharpe_stats,
            "Min Volatility": min_vol_stats
        }

        # --- 3. Calculate Individual Asset Stats ---
        print("Calculating Individual Asset Stats...")
        individual_assets = {}
        for ticker in TICKERS:
            weights = np.zeros(len(TICKERS))
            weights[TICKERS.index(ticker)] = 1.0
            asset_stats = get_portfolio_stats(weights, mean_returns, cov_matrix)
            individual_assets[ticker] = asset_stats
            
        # --- 4. Calculate Sample Model Portfolios (for Analysis) ---
        print("Calculating Sample Portfolios...")
        num_assets = len(TICKERS)
        sample_portfolios = {} # Initialize dictionary

        # Model 1: Equally Weighted (1/n)
        equal_weights = np.array([1 / num_assets] * num_assets)
        sample_portfolios["Equally Weighted (1/n)"] = get_portfolio_stats(
            equal_weights, mean_returns, cov_matrix
        )

        # Model 2: 100% in Highest Return Asset
        max_ret_ticker = mean_returns.idxmax()
        sample_portfolios[f"100% {max_ret_ticker} (Max Ret)"] = individual_assets[max_ret_ticker]

        # Model 3: 100% in Lowest Risk Asset
        cov_variances = pd.Series(np.diag(cov_matrix), index=TICKERS)
        min_risk_ticker = cov_variances.idxmin()
        sample_portfolios[f"100% {min_risk_ticker} (Min Risk)"] = individual_assets[min_risk_ticker]
        
        # --- 5. Calculate the Efficient Frontier ---
        print("Calculating Efficient Frontier...")
        frontier_returns, frontier_volatilities = calculate_efficient_frontier(
            mean_returns, cov_matrix
        )
        frontier_data = (frontier_volatilities, frontier_returns)
        
        # --- 6. Plot All Results ---
        # *** LINHA CORRIGIDA ***
        plot_results(
            frontier_data=frontier_data,
            optimal_portfolios=optimal_portfolios,
            sample_portfolios=sample_portfolios, # Corrigido de 'sample_postfolios'
            individual_assets=individual_assets
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    pd.set_option('display.width', 100)
    pd.set_option('display.precision', 4)
    run_analysis()