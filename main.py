import pandas as pd
import numpy as np

# Import your modules
from portfolio_optimizer.data import get_annualized_inputs
from portfolio_optimizer.model import get_portfolio_stats
from portfolio_optimizer.optimization import (
    find_max_sharpe_portfolio,
    find_min_volatility_portfolio,
    calculate_efficient_frontier,
    # Import the naive optimizer
    find_max_naive_sharpe_portfolio
)
from portfolio_optimizer.plots import plot_results

# A 10-asset basket for robust diversification
TICKERS = [
    # US Equities
    'SPY',  # S&P 500
    'QQQ',  # Nasdaq 100
    'IWM',  # US Small-Cap
    
    # International Equities
    'EFA',  # Developed Markets (ex-US)
    'EEM',  # Emerging Markets
    
    # Fixed Income
    'TLT',  # 20+ Year US Treasury Bonds
    
    # Real Assets & Commodities
    'VNQ',  # US Real Estate
    'GLD',  # Gold
    'DBC',  # Broad Commodities
    'XLE'   # Energy Sector
]
START_DATE = '2000-01-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.02 # 2%

# --- Main analysis function ---
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
        
        # --- 2. Calculate Optimal Portfolios (Markowitz) ---
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
            
        # --- 4. Calculate Sample Model Portfolios (for 'X' markers) ---
        print("Calculating Sample Portfolios...")
        num_assets = len(TICKERS)
        sample_portfolios = {} 

        # Model 1: Equally Weighted (1/n)
        equal_weights = np.array([1 / num_assets] * num_assets)
        sample_portfolios["Equally Weighted (1/n)"] = get_portfolio_stats(
            equal_weights, mean_returns, cov_matrix
        )
        
        # Model 2: Calculate the Naive Sharpe Portfolio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        naive_max_sharpe_weights = find_max_naive_sharpe_portfolio(
            mean_returns, individual_vols, RISK_FREE_RATE
        )
        naive_max_sharpe_stats = get_portfolio_stats(
            naive_max_sharpe_weights, mean_returns, cov_matrix
        )
        sample_portfolios["Naive Max Sharpe (No Cov)"] = naive_max_sharpe_stats
        
        # Model 3: 100% in Highest Return Asset
        max_ret_ticker = mean_returns.idxmax()
        max_ret_weights = np.zeros(len(TICKERS))
        max_ret_weights[TICKERS.index(max_ret_ticker)] = 1.0
        sample_portfolios[f"100% {max_ret_ticker} (Max Ret)"] = individual_assets[max_ret_ticker]

        # Model 4: 100% in Lowest Risk Asset
        cov_variances = pd.Series(np.diag(cov_matrix), index=TICKERS)
        min_risk_ticker = cov_variances.idxmin()
        min_risk_weights = np.zeros(len(TICKERS))
        min_risk_weights[TICKERS.index(min_risk_ticker)] = 1.0
        sample_portfolios[f"100% {min_risk_ticker} (Min Risk)"] = individual_assets[min_risk_ticker]
        
        # --- 5. Calculate the Efficient Frontier ---
        print("Calculating Efficient Frontier...")
        # We no longer need the frontier_weights
        frontier_returns, frontier_volatilities = calculate_efficient_frontier(
            mean_returns, cov_matrix
        )
        frontier_data = (frontier_volatilities, frontier_returns)
        
        # --- 6. Create Composition Dictionary ---
        # This dict is for your original table
        portfolio_compositions = {
            "Max Sharpe Ratio": max_sharpe_weights,
            "Min Volatility": min_vol_weights,
            "Naive Max Sharpe (No Cov)": naive_max_sharpe_weights,
            "Equally Weighted (1/n)": equal_weights,
            f"100% {max_ret_ticker} (Max Ret)": max_ret_weights,
            f"100% {min_risk_ticker} (Min Risk)": min_risk_weights
        }
        
        # --- 7. Plot All Results ---
        print("\nDisplaying results plot...")
        plot_results(
            frontier_data=frontier_data,
            optimal_portfolios=optimal_portfolios,
            sample_portfolios=sample_portfolios,
            individual_assets=individual_assets,
            # Pass only the original composition table
            portfolio_compositions=portfolio_compositions,
            tickers=TICKERS
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    pd.set_option('display.width', 100)
    pd.set_option('display.precision', 4)
    run_analysis()