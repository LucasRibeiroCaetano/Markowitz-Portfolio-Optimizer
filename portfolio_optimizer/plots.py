import matplotlib.pyplot as plt
import matplotx
import numpy as np
import pandas as pd # Import pandas for formatting

def plot_results(
    frontier_data: tuple[np.ndarray, np.ndarray],
    optimal_portfolios: dict[str, tuple[float, float]],
    sample_portfolios: dict[str, tuple[float, float]],
    individual_assets: dict[str, tuple[float, float]],
    # Arguments for composition
    portfolio_compositions: dict[str, np.ndarray],
    tickers: list[str]
):
    """
    Plots the Efficient Frontier, optimal portfolios, sample
    portfolios, and individual assets on a single chart.
    Also displays the composition of key portfolios in two columns.
    """
    
    # 1. Get the data from the inputs
    frontier_volatilities, frontier_returns = frontier_data
    
    # 2. Apply the 'matplotx' and 'nord' style
    try:
        plt.style.use(matplotx.styles.nord)
    except:
        plt.style.use("ggplot") # Fallback style
    
    # 3. Create the plot - Make it wider for the two columns
    fig, ax = plt.subplots(figsize=(18, 9))
    
    # 4. Adjust layout to make space on the right
    # Use 60% of the figure for the plot, leaving 40% on the right
    plt.subplots_adjust(left=0.08, right=0.60, top=0.9, bottom=0.1)
    
    # 5. Plot the Efficient Frontier
    ax.plot(
        frontier_volatilities, 
        frontier_returns, 
        linestyle='--', 
        label='Efficient Frontier'
    )
    
    # 6. Plot Individual Assets (as dots)
    asset_vols = [v[0] for v in individual_assets.values()]
    asset_rets = [v[1] for v in individual_assets.values()]
    ax.scatter(
        asset_vols, 
        asset_rets, 
        marker='o', 
        s=50, 
        color='grey',
        label='Individual Assets'
    )

    # 7. Plot Sample Portfolios (as 'X's)
    for name, (vol, ret) in sample_portfolios.items():
        ax.scatter(
            vol, 
            ret, 
            marker='X', 
            s=150, 
            label=f'Sample: {name}'
        )

    # 8. Plot Optimal Portfolios (as stars)
    for name, (vol, ret) in optimal_portfolios.items():
        ax.scatter(
            vol, 
            ret, 
            marker='*', 
            s=250,
            label=f'Optimal: {name}'
        )
    
    # 9. Final plot styling
    ax.set_title('Markowitz Portfolio Optimization', fontsize=16)
    ax.set_xlabel('Annualized Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    
    # Use a standard matplotlib legend (placed on the plot)
    ax.legend(loc='lower right')
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    
    # --- 10. NEW: Two-Column Composition Table ---
    
    # Define X and Y coordinates for the two columns
    # (1, 1) is top-right corner of the *axes*
    x_col_1 = 1.05  # X-position for first column
    x_col_2 = 1.30  # X-position for second column
    
    y_col_1_tracker = 1.0 # Y-position for first column
    y_col_2_tracker = 1.0 # Y-position for second column
    
    # Calculate how many items per column (for 5 items, this is 3)
    items_in_col_1 = (len(portfolio_compositions) + 1) // 2

    # Title for the composition box
    ax.text(x_col_1, y_col_1_tracker, "Portfolio Compositions",
            transform=ax.transAxes, fontsize=14,
            fontweight='bold', ha='left')
    
    # Align the starting Y for both columns
    y_col_1_tracker -= 0.08
    y_col_2_tracker = y_col_1_tracker
    
    # Loop through all portfolios
    for i, (name, weights) in enumerate(portfolio_compositions.items()):
        
        # Decide which column (X) and Y-tracker to use
        if i < items_in_col_1:
            x_base = x_col_1
            current_y = y_col_1_tracker
        else:
            x_base = x_col_2
            current_y = y_col_2_tracker

        # Add portfolio title
        ax.text(x_base, current_y, f"{name}:",
                transform=ax.transAxes, fontsize=11,
                fontweight='bold', ha='left')
        current_y -= 0.05 # Move down

        # Format weights (filter for > 1% to keep it clean)
        comp = pd.DataFrame({'Weight': weights}, index=tickers)
        comp = comp[comp['Weight'] > 0.01] # Filter 1%
        comp = comp.sort_values(by='Weight', ascending=False)
        
        if comp.empty:
            ax.text(x_base + 0.01, current_y, "  (No assets > 1%)",
                    transform=ax.transAxes, fontsize=10,
                    ha='left', style='italic')
            current_y -= 0.05
        else:
            for ticker, row in comp.iterrows():
                # Format: "  - AAPL: 20.5%"
                weight_str = f"{row['Weight']:,.1%}"
                line = f"  - {ticker}: {weight_str}"
                
                ax.text(x_base + 0.01, current_y, line,
                        transform=ax.transAxes, fontsize=10,
                        ha='left')
                current_y -= 0.04 # Move down for next line
        
        current_y -= 0.03 # Extra space before next portfolio
        
        # Update the Y-tracker for the column we just used
        if i < items_in_col_1:
            y_col_1_tracker = current_y
        else:
            y_col_2_tracker = current_y

    # 11. Final show
    plt.show()