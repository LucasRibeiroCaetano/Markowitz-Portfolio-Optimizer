import matplotlib.pyplot as plt
import matplotx
import numpy as np

def plot_results(
    frontier_data: tuple[np.ndarray, np.ndarray],
    optimal_portfolios: dict[str, tuple[float, float]],
    sample_portfolios: dict[str, tuple[float, float]],
    individual_assets: dict[str, tuple[float, float]]
):
    """
    Plots the Efficient Frontier, optimal portfolios, sample
    portfolios, and individual assets on a single chart.
    """
    
    # 1. Get the data from the inputs
    frontier_volatilities, frontier_returns = frontier_data
    
    # 2. Apply the 'matplotx' and 'nord' style
    # This gives the clean look you wanted
    plt.style.use(matplotx.styles.nord)
    
    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 4. Plot the Efficient Frontier
    ax.plot(
        frontier_volatilities, 
        frontier_returns, 
        linestyle='--', 
        label='Efficient Frontier'
    )
    
    # 5. Plot Individual Assets (as dots)
    # We unpack the dictionary items
    for ticker, (vol, ret) in individual_assets.items():
        ax.scatter(
            vol, 
            ret, 
            marker='o', 
            s=50, 
            label=f'Individual: {ticker}'
        )

    # 6. Plot Sample Portfolios (as 'X's)
    for name, (vol, ret) in sample_portfolios.items():
        ax.scatter(
            vol, 
            ret, 
            marker='X', 
            s=150, 
            label=f'Sample: {name}'
        )

    # 7. Plot Optimal Portfolios (as stars)
    for name, (vol, ret) in optimal_portfolios.items():
        ax.scatter(
            vol, 
            ret, 
            marker='*', 
            s=250,  # Larger
            label=f'Optimal: {name}'
        )
    
    # 8. Final plot styling
    ax.set_title('Markowitz Portfolio Optimization', fontsize=16)
    ax.set_xlabel('Annualized Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    
    # Use a standard matplotlib legend
    ax.legend(loc='best')
    
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Finally, display the plot
    print("Displaying results plot...")
    plt.show()