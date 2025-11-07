import pandas as pd
from portfolio_optimizer.data import get_annualized_inputs

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
        # This calls the module you just created
        mean_returns, cov_matrix = get_annualized_inputs(
            tickers=TICKERS,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        print("--- Annualized Mean Returns ---")
        print(mean_returns)
        print("\n--- Annualized Covariance Matrix ---")
        print(cov_matrix)
        
        # --- NEXT STEPS WILL GO HERE ---
        # 2. Calculate core model stats (to be built)
        # 3. Run optimization (to be built)
        # 4. Calculate sample portfolios (to be built)
        # 5. Plot results (to be built)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # This ensures the code runs only when the script is executed directly
    pd.set_option('display.width', 100) # For better terminal printing
    pd.set_option('display.precision', 4)
    run_analysis()