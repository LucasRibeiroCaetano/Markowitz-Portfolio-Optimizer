import numpy as np
import scipy.optimize as sco
from pandas import DataFrame, Series

# Import from the other .py files in the same package
from .model import get_portfolio_stats, get_negative_sharpe_ratio

def find_max_sharpe_portfolio(
    mean_returns: Series, 
    cov_matrix: DataFrame, 
    risk_free_rate: float
) -> np.ndarray:
    """
    Finds the optimal portfolio weights for the Maximum Sharpe Ratio.
    """
    num_assets = len(mean_returns)
    
    # Arguments for the optimizer
    # We are minimizing the 'get_negative_sharpe_ratio' function
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: 0 <= weight <= 1 for each asset (no short-selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array(num_assets * [1. / num_assets])

    # Run the optimization
    # 'SLSQP' is a method good for this type of constrained problem
    result = sco.minimize(
        get_negative_sharpe_ratio,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x # 'x' is the array of optimal weights


def find_min_volatility_portfolio(
    mean_returns: Series, 
    cov_matrix: DataFrame
) -> np.ndarray:
    """
    Finds the optimal portfolio weights for the Global Minimum Volatility.
    """
    num_assets = len(mean_returns)
    
    # Define the function to minimize: portfolio volatility
    # We pass 'weights' as the first argument
    def get_volatility(weights, mean_returns, cov_matrix):
        # We only need the second return value (volatility)
        return get_portfolio_stats(weights, mean_returns, cov_matrix)[1]

    # Arguments for the optimizer
    args = (mean_returns, cov_matrix)
    
    # Constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: 0 <= weight <= 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array(num_assets * [1. / num_assets])

    # Run the optimization
    result = sco.minimize(
        get_volatility,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


def calculate_efficient_frontier(
    mean_returns: Series, 
    cov_matrix: DataFrame, 
    num_portfolios: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the efficient frontier by optimizing for minimum
    volatility at a range of target returns.
    """
    num_assets = len(mean_returns)
    
    # --- Find the min and max returns for the frontier range ---
    # 1. Get the weights for the min volatility portfolio
    min_vol_weights = find_min_volatility_portfolio(mean_returns, cov_matrix)
    # 2. Get the return of that portfolio
    min_vol_return, _ = get_portfolio_stats(min_vol_weights, mean_returns, cov_matrix)
    
    # 3. We'll use the highest single-asset return as our max
    max_return = mean_returns.max()
    
    # Generate a range of target returns to optimize for
    target_returns = np.linspace(min_vol_return, max_return, num_portfolios)
    
    frontier_volatilities = []

    # Define the objective function (minimize volatility)
    def get_volatility(weights, mean_returns, cov_matrix):
        return get_portfolio_stats(weights, mean_returns, cov_matrix)[1]

    # Define the base bounds and initial guess
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    # --- Loop through each target return and find the min risk ---
    for target_return in target_returns:
        # We add a *new* constraint for the target return
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, # Sum = 1
            {'type': 'eq', 'fun': lambda w: get_portfolio_stats(w, mean_returns, cov_matrix)[0] - target_return} # Return = target
        ]
        
        result = sco.minimize(
            get_volatility,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            frontier_volatilities.append(result.fun) # result.fun is the volatility
        else:
            # If optimization fails at a point, just use nan
            frontier_volatilities.append(np.nan)

    return np.array(target_returns), np.array(frontier_volatilities)