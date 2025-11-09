import numpy as np
import scipy.optimize as sco
from pandas import DataFrame, Series 

from .model import get_portfolio_stats, get_negative_sharpe_ratio

def find_max_sharpe_portfolio(
    mean_returns: Series, 
    cov_matrix: DataFrame, 
    risk_free_rate: float
) -> np.ndarray:
    """
    Finds the optimal portfolio weights for the Maximum Sharpe Ratio.
    (This is the correct Markowitz method)
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])

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
    return result.x


def find_min_volatility_portfolio(
    mean_returns: Series, 
    cov_matrix: DataFrame
) -> np.ndarray:
    """
    Finds the optimal portfolio weights for the Global Minimum Volatility.
    (This is the correct Markowitz method)
    """
    num_assets = len(mean_returns)
    
    def get_volatility(weights, mean_returns, cov_matrix):
        return get_portfolio_stats(weights, mean_returns, cov_matrix)[1]

    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])

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
    Calculates the efficient frontier.
    (This is the correct Markowitz method)
    """
    num_assets = len(mean_returns)
    
    min_vol_weights = find_min_volatility_portfolio(mean_returns, cov_matrix)
    min_vol_return, _ = get_portfolio_stats(min_vol_weights, mean_returns, cov_matrix)
    
    max_return = mean_returns.max()
    
    target_returns = np.linspace(min_vol_return, max_return, num_portfolios)
    
    frontier_volatilities = []

    def get_volatility(weights, mean_returns, cov_matrix):
        return get_portfolio_stats(weights, mean_returns, cov_matrix)[1]

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    for target_return in target_returns:
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
            frontier_volatilities.append(result.fun)
        else:
            frontier_volatilities.append(np.nan)

    # --- Return only returns and volatilities ---
    return np.array(target_returns), np.array(frontier_volatilities)

def get_negative_naive_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: Series,
    individual_volatilities: np.ndarray,
    risk_free_rate: float
) -> float:
    """
    Calculates a "naive" Sharpe Ratio that ignores correlation.
    """
    portfolio_return = np.dot(weights.T, mean_returns)
    
    # Risk is calculated naively as a weighted average
    naive_volatility = np.dot(weights.T, individual_volatilities)
    
    if naive_volatility == 0:
        return 0.0 if portfolio_return == risk_free_rate else -np.inf
        
    sharpe_ratio = (portfolio_return - risk_free_rate) / naive_volatility
    
    return -sharpe_ratio


def find_max_naive_sharpe_portfolio(
    mean_returns: Series,
    individual_volatilities: np.ndarray,
    risk_free_rate: float
) -> np.ndarray:
    """
    Finds the optimal portfolio weights for the "Naive" Sharpe Ratio
    (which ignores covariance).
    """
    num_assets = len(mean_returns)
    args = (mean_returns, individual_volatilities, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])

    result = sco.minimize(
        get_negative_naive_sharpe_ratio,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        raise RuntimeError(f"Naive optimization failed: {result.message}")

    return result.x