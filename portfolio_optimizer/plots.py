import matplotlib.pyplot as plt
import matplotx # Import matplotx
import numpy as np
import pandas as pd # Import pandas for formatting

# --- NEW: Helper function to plot pie charts ---
def _plot_composition_pies(fig, compositions, tickers_list):
    """
    Internal helper function to draw the "Key Portfolios"
    in a clean two-column layout, as pie charts.
    """
    
    # Convert tickers list to a numpy array for easy masking
    tickers = np.array(tickers_list)
    
    # --- Define the 2-Column Layout ---
    items_in_col_1 = (len(compositions) + 1) // 2 # 3 items in col 1
    
    x_col_1 = 0.58  # X-position for first column
    x_col_2 = 0.78  # X-position for second column
    
    y_col_1_tracker = 0.90 # Start Y for col 1
    y_col_2_tracker = 0.90 # Start Y for col 2
    
    pie_width = 0.2  # Width of each pie
    pie_height = 0.2 # Height of each pie
    y_padding = 0.12  # Vertical padding between pies
    
    # Get the default color cycle from the style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (name, weights) in enumerate(compositions.items()):
        
        # --- 1. Decide position ---
        if i < items_in_col_1:
            x_base = x_col_1
            current_y = y_col_1_tracker
        else:
            x_base = x_col_2
            current_y = y_col_2_tracker
            
        # --- 2. Filter data for the pie ---
        # We only plot slices > 1% to keep it clean
        mask = weights > 0.01
        sizes = weights[mask]
        
        # If no slices are > 1%, use a 0.01% filter
        if sizes.sum() == 0:
             mask = weights > 0.0001
             sizes = weights[mask]
             if sizes.sum() == 0:
                 continue # Skip if truly empty
            
        labels = tickers[mask]
        
        # --- 3. Create the new Axes for the pie ---
        # Coordinates are [left, bottom, width, height]
        pie_ax = fig.add_axes([
            x_base, 
            current_y - pie_height, 
            pie_width, 
            pie_height
        ])
        
        # --- 4. Plot the pie chart ---
        pie_ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%', # Format as percentage
            textprops={'fontsize': 8},
            colors=colors # Use the style's colors
        )
        pie_ax.set_title(name, fontsize=10, y=1.05, fontweight='bold')      

        # --- 5. Update Y-tracker for the next pie ---
        if i < items_in_col_1:
            y_col_1_tracker -= (pie_height + y_padding)
        else:
            y_col_2_tracker -= (pie_height + y_padding)


# --- MODIFIED PLOT_RESULTS FUNCTION ---
def plot_results(
    frontier_data: tuple[np.ndarray, np.ndarray],
    optimal_portfolios: dict[str, tuple[float, float]],
    sample_portfolios: dict[str, tuple[float, float]],
    individual_assets: dict[str, tuple[float, float]],
    portfolio_compositions: dict[str, np.ndarray],
    tickers: list[str]
):
    """
    Plots the Efficient Frontier and overlays the
    Key Portfolio Compositions as a stack of pie charts.
    """
    
    # 1. Get the data from the inputs
    frontier_volatilities, frontier_returns = frontier_data
    
    # 2. Apply the 'github[dark]' style (using your correct syntax)
    try:
        plt.style.use(matplotx.styles.github["dark"])
    except Exception:
        plt.style.use("ggplot") # Fallback style
    
    # 3. Create the plot - Wide for two columns
    fig, ax = plt.subplots(figsize=(20, 12)) # 20 wide, 12 tall
    
    # 4. Adjust layout to make space on the right
    # Use 55% of the figure for the plot, leaving 45% on the right
    plt.subplots_adjust(left=0.07, right=0.55, top=0.9, bottom=0.1)
    
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
    
    # Set legend to bottom right
    ax.legend(loc='lower right')
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    
    # --- 10. PLOT KEY COMPOSITION PIE CHARTS ---
    # We pass the main 'fig' object, not 'ax'
    _plot_composition_pies(
        fig, portfolio_compositions, tickers
    )

    # 11. Final show
    plt.show()