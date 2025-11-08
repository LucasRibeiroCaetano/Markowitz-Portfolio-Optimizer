# Markowitz Portfolio Optimizer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20SciPy-orange)
![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20matplotx-brightgreen)

A Python application that implements Markowitz's Modern Portfolio Theory (MPT). This tool fetches historical stock data, calculates the Efficient Frontier, and identifies optimal portfolios based on user-defined risk preferences.

The project demonstrates a clean, modular code structure, separating concerns for data handling, mathematical modeling, optimization, and visualization.


*(Add a screenshot of your final plot here)*

## Key Features

* **Efficient Frontier Calculation**: Generates the complete set of optimal portfolios by running a constrained optimization for a range of target returns.
* **Optimal Portfolio Identification**: Uses `scipy.optimize` to find the weights for:
    * **Maximum Sharpe Ratio Portfolio** (The "tangent portfolio")
    * **Global Minimum Volatility Portfolio** (The safest possible portfolio)
* **Comparative Analysis**: Plots the optimal portfolios against "naive" strategies (e.g., "Equally Weighted") and individual assets for clear performance comparison.
* **Modern Visualization**: Uses `matplotx` and the `matplotlib-nord` theme for a clean, publication-ready plot.
* **Practical Constraints**: The optimization is run with realistic constraints:
    * No short-selling (weights $\ge$ 0)
    * Fully invested (sum of weights = 1)

## Technology Stack

* **Data Retrieval**: `yfinance`
* **Data Analysis**: `pandas` & `numpy`
* **Optimization**: `scipy` (specifically `scipy.optimize.minimize` with 'SLSQP')
* **Visualization**: `matplotlib`, `matplotx`, `matplotlib-nord`

## Project Structure

The project uses a modular structure to separate concerns, making the code clean, maintainable, and testable.

```ascii
markowitz_optimizer/
│
├──  portfolio_optimizer/
│   ├── __init__.py
│   ├── data.py         # Handles data fetching and processing
│   ├── model.py        # Core mathematical functions (portfolio stats, sharpe)
│   ├── optimization.py # SciPy optimization logic for finding portfolios
│   └── plots.py        # Visualization logic
│
├── main.py             # Main entry point to run the analysis
├── requirements.txt
└── README.md
```

## Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/markowitz-optimizer.git](https://github.com/your-username/markowitz-optimizer.git)
    cd markowitz-optimizer
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the analysis:**

    ```bash
    python main.py
    ```

    This will fetch the latest data, perform all calculations, and display the final optimization plot.

## Customization

To analyze different assets or timeframes, simply modify the global constants in `main.py`:

```python
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.02 
```