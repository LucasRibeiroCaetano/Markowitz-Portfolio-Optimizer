import numpy as np
from pandas import Series, DataFrame

# Note: We use type hints for clarity.
# A 'weights' array is a 1D numpy array.
# Mean returns is a Series (or 1D array).
# Covariance matrix is a DataFrame (or 2D array).

def get_portfolio_stats(
    weights: np.ndarray,
    mean_returns: Series | np.ndarray,
    cov_matrix: DataFrame | np.ndarray
) -> tuple[float, float]:
    """
    Calculates the annualized portfolio return and volatility.
    """
    # Convert pandas objects to numpy arrays if necessary
    # This ensures pure numpy operations
    mean_returns_arr = np.asarray(mean_returns)
    cov_matrix_arr = np.asarray(cov_matrix)
    
    # Calculate portfolio return
    # E(Rp) = w^T * E(R)
    portfolio_return = np.dot(weights.T, mean_returns_arr)

    # Calculate portfolio variance
    # Var(Rp) = w^T * Cov * w
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_arr, weights))
    
    # Calculate portfolio volatility (standard deviation)
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    return portfolio_return, portfolio_volatility


def get_negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: Series | np.ndarray,
    cov_matrix: DataFrame | np.ndarray,
    risk_free_rate: float
) -> float:
    """
    Calculates the negative Sharpe Ratio for optimization.
    
    We return the *negative* ratio because scipy's 'minimize'
    function finds the minimum value of a function. Minimizing
    the negative Sharpe Ratio is equivalent to maximizing the
    positive Sharpe Ratio.
    """
    portfolio_return, portfolio_volatility = get_portfolio_stats(
        weights, mean_returns, cov_matrix
    )
    
    # Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
    # Handle the case where volatility is zero to avoid division error
    if portfolio_volatility == 0:
        # Returning a large negative number (bad sharpe) if no risk
        # or 0 if return is also 0.
        return 0.0 if portfolio_return == risk_free_rate else -np.inf
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Return the negative for the optimizer
    return -sharpe_ratio