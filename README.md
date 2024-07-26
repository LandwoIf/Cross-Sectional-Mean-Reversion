# Cross-Sectional Mean Reversion Strategy for Cryptocurrencies
============================================================

This code implements a cross-sectional mean reversion strategy for cryptocurrencies traded on MEXC. The strategy aims to profit from mid-term price reversals while maintaining market neutrality from Ernie Chan's 'Algorithmic Trading: Winning Strategies and Their Rationale'.

## Key Features
- Uses top 20 cryptocurrencies by 24-hour trading volume
- Implements both return-based selection and volatility filtering
- Assumes zero trading fees (as per MEXC's conditions)
- Backtest framework using Backtrader

**DISCLAIMER:** This code is for educational purposes only. Always perform your own due diligence before trading. Past performance does not guarantee future results.

## Getting Started

### Prerequisites
- Python 3.x
- Pandas
- NumPy
- Backtrader

### Installation
```sh
pip install pandas numpy backtrader
```

### Usage
1. Prepare the historical data for the top 20 cryptocurrencies by 24-hour trading volume and save it as `mexc_data.csv`.
2. Customize the `create_bt_data` function to fetch historical data for each cryptocurrency.
3. Run the script to perform a backtest.

### Example
```python
if __name__ == "__main__":
    # Load data for the top 20 cryptocurrencies by 24h volume
    crypto_data = pd.read_csv('mexc_data.csv')  # Assume we've saved the data to a CSV
    top_20_cryptos = crypto_data.sort_values('24h Volume', ascending=False).head(20)

    # Function to create Backtrader data feeds
    def create_bt_data(symbol, start_date, end_date):
        # This function would need to fetch historical data for each cryptocurrency
        # For this example, we'll use random data
        return bt.feeds.YahooFinanceData(dataname=symbol, fromdate=start_date, todate=end_date)

    # Create Backtrader data feeds
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 7, 26)
    datas = [create_bt_data(row['Pair'], start_date, end_date) for _, row in top_20_cryptos.iterrows()]

    # Run the backtest
    dd, cagr, sharpe = backtest(datas, CryptoCSMR, plot=True, n=10)
    print(f"Max Drawdown: {dd:.2f}%")
    print(f"Annual Return: {cagr:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
```

## Conclusion and Next Steps
This code provides a framework for implementing a cross-sectional mean reversion strategy for cryptocurrencies. However, before using it for live trading, consider the following steps:

1. **Data Source:** Implement a proper data fetching mechanism for historical cryptocurrency data from MEXC or a reliable data provider.
2. **~~Optimization~~ Overfitting:** Experiment with different parameters (e.g., lookback periods, number of cryptocurrencies) to optimize the strategy's performance.
3. **Robustness Testing:** Perform out-of-sample testing and consider Monte Carlo simulations to assess the strategy's robustness.
4. **Market Regimes:** Analyze the strategy's performance under different market conditions and consider implementing regime-switching mechanisms.
