import pandas as pd
import numpy as np
import argparse

# Import your modules
from portfolio_optimizer.data import get_annualized_inputs
from portfolio_optimizer.model import get_portfolio_stats
from portfolio_optimizer.optimization import (
    find_max_sharpe_portfolio,
    find_min_volatility_portfolio,
    calculate_efficient_frontier,
    find_max_naive_sharpe_portfolio
)
from portfolio_optimizer.plots import plot_results

# --- Define defaults ---
DEFAULT_TICKERS = [
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT',
    'BTC-USD', 'GLD', 'DBC', 'XLE'
]
DEFAULT_START_DATE = '2000-01-01'
DEFAULT_END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.02 # This remains a global constant

# --- Function to parse command-line arguments ---
def parse_arguments():
    """
    Parses command-line arguments for tickers, start date, and end date.
    """
    parser = argparse.ArgumentParser(
        description="Markowitz Portfolio Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-t', '--tickers',
        nargs='+', # Accepts one or more tickers
        default=DEFAULT_TICKERS,
        help=f"Up to 20 asset tickers to analyze."
    )
    
    parser.add_argument(
        '-s', '--start',
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date in YYYY-MM-DD format."
    )
    
    parser.add_argument(
        '-e', '--end',
        type=str,
        default=DEFAULT_END_DATE,
        help="End date in YYYY-MM-DD format."
    )
    
    args = parser.parse_args()
    
    # --- NEW: Validate the 20-asset limit ---
    if len(args.tickers) > 20:
        parser.error(f"Maximum of 20 tickers allowed. You provided {len(args.tickers)}.")
        
    # Convert tickers to uppercase
    args.tickers = [t.upper() for t in args.tickers]
    
    return args

# --- Main analysis function ---
def run_analysis(tickers, start_date, end_date):
    """
    Runs the full portfolio analysis based on the provided arguments.
    """
    print("Starting Portfolio Analysis...")
    print(f"Fetching data for: {', '.join(tickers)}\n")
    
    try:
        # 1. Get data
        mean_returns, cov_matrix = get_annualized_inputs(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
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
        for ticker in tickers:
            weights = np.zeros(len(tickers))
            weights[tickers.index(ticker)] = 1.0
            asset_stats = get_portfolio_stats(weights, mean_returns, cov_matrix)
            individual_assets[ticker] = asset_stats
            
        # --- 4. Calculate Sample Model Portfolios (for 'X' markers) ---
        print("Calculating Sample Portfolios...")
        num_assets = len(tickers)
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
        sample_portfolios["Max Sharpe (No Cov)"] = naive_max_sharpe_stats
        
        # Model 3: 100% in Highest Return Asset
        max_ret_ticker = mean_returns.idxmax()
        max_ret_weights = np.zeros(len(tickers))
        max_ret_weights[tickers.index(max_ret_ticker)] = 1.0
        sample_portfolios["Max Return"] = individual_assets[max_ret_ticker]

        # Model 4: 100% in Lowest Risk Asset
        cov_variances = pd.Series(np.diag(cov_matrix), index=tickers)
        min_risk_ticker = cov_variances.idxmin()
        min_risk_weights = np.zeros(len(tickers))
        min_risk_weights[tickers.index(min_risk_ticker)] = 1.0
        sample_portfolios["Min Risk"] = individual_assets[min_risk_ticker]
        
        # --- 5. Calculate the Efficient Frontier ---
        print("Calculating Efficient Frontier...")
        frontier_returns, frontier_volatilities = calculate_efficient_frontier(
            mean_returns, cov_matrix
        )
        frontier_data = (frontier_volatilities, frontier_returns)
        
        # --- 6. Create Composition Dictionary ---
        portfolio_compositions = {
            "Max Sharpe Ratio": max_sharpe_weights,
            "Min Volatility": min_vol_weights,
            "Max Sharpe (No Cov)": naive_max_sharpe_weights,
            "Max Return": max_ret_weights,
            "Min Risk": min_risk_weights
        }
        
        # --- 7. Plot All Results ---
        print("\nDisplaying results plot...")
        plot_results(
            frontier_data=frontier_data,
            optimal_portfolios=optimal_portfolios,
            sample_portfolios=sample_portfolios,
            individual_assets=individual_assets,
            portfolio_compositions=portfolio_compositions,
            tickers=tickers # Pass the dynamic tickers
        )

    except Exception as e:
        print(f"An error occurred: {e}")

# --- MODIFIED: Main execution block ---
if __name__ == "__main__":
    pd.set_option('display.width', 100)
    pd.set_option('display.precision', 4)
    
    # 1. Parse arguments from the command line
    args = parse_arguments()
    
    # 2. Run the analysis with the parsed arguments
    run_analysis(args.tickers, args.start, args.end)