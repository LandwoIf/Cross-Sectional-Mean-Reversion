# Cross-Sectional Mean Reversion Strategy for Cryptocurrencies
# ============================================================
#
# This code implements a cross-sectional mean reversion strategy for cryptocurrencies
# traded on MEXC. The strategy aims to profit from mid-term price reversals while
# maintaining market neutrality.
#
# Key features:
# - Uses top 20 cryptocurrencies by 24-hour trading volume
# - Implements both return-based selection and volatility filtering
# - Assumes zero trading fees (as per MEXC's conditions)
# - Backtest framework using Backtrader
#
# DISCLAIMER: This code is for educational purposes only. Always perform your own
# due diligence before trading. Past performance does not guarantee future results.

# Import required libraries
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

# Helper functions for array manipulation
def min_n(array, n):
    """Return indices of n smallest elements in array."""
    return np.argpartition(array, n)[:n]

def max_n(array, n):
    """Return indices of n largest elements in array."""
    return np.argpartition(array, -n)[-n:]

# Define the main strategy class
class CryptoCSMR(bt.Strategy):
    """
    Cryptocurrency Cross-Sectional Mean Reversion Strategy
    
    This strategy selects cryptocurrencies based on their deviation from
    the mean market return and their recent volatility.
    """
    
    params = (
        ('n', 20),  # Number of cryptocurrencies to trade
    )
    
    def __init__(self):
        self.inds = {}
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]["pct"] = bt.indicators.PercentChange(d.close, period=1)
            self.inds[d]["std"] = bt.indicators.StandardDeviation(d.close, period=5)

    def next(self):
        # Filter for available data
        available = list(filter(lambda d: len(d) > 5, self.datas))
        if len(available) < self.p.n:
            return  # Not enough data yet
        
        # Calculate returns and standard deviations
        rets = np.zeros(len(available))
        stds = np.zeros(len(available))
        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]
            stds[i] = self.inds[d]['std'][0]

        # Calculate market return and deviations
        market_ret = np.mean(rets)
        weights = -(rets - market_ret)
        
        # Select cryptocurrencies based on return deviation and volatility
        max_weights_index = max_n(np.abs(weights), self.p.n)
        low_volatility_index = min_n(stds, self.p.n)
        selected_weights_index = np.intersect1d(max_weights_index, low_volatility_index)
        
        if not len(selected_weights_index):
            return  # No good trades today
            
        # Normalize weights
        selected_weights = weights[selected_weights_index]
        weights = weights / np.sum(np.abs(selected_weights))      
        
        # Place orders
        for i, d in enumerate(available):
            if i in selected_weights_index:
                self.order_target_percent(d, target=weights[i])
            else:
                self.order_target_percent(d, 0)

# Backtesting function
def backtest(datas, strategy, plot=False, **kwargs):
    """
    Run a backtest of the given strategy on the provided data.
    
    Args:
    datas: List of data feeds
    strategy: Strategy class to backtest
    plot: Boolean, whether to generate a plot of results
    **kwargs: Additional keyword arguments for the strategy
    
    Returns:
    Tuple of (max drawdown, annualized return, Sharpe ratio)
    """
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_cash(100000)  # Starting with $100,000
    cerebro.broker.setcommission(commission=0.0)  # Set commission to 0
    for data in datas:
        cerebro.adddata(data)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addstrategy(strategy, **kwargs)
    results = cerebro.run()
    if plot:
        cerebro.plot(iplot=False)[0][0]
    return (results[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            results[0].analyzers.returns.get_analysis()['rnorm100'],
            results[0].analyzers.sharperatio.get_analysis()['sharperatio'])

# Main execution
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

# Conclusion and Next Steps
# =========================
#
# This code provides a framework for implementing a cross-sectional mean reversion
# strategy for cryptocurrencies. However, before using it for live trading, consider
# the following steps:
#
# 1. Data Source: Implement a proper data fetching mechanism for historical
#    cryptocurrency data from MEXC or a reliable data provider.
#
# 2. Risk Management: Add additional risk management features such as stop-loss
#    orders and position sizing based on volatility.
#
# 3. Transaction Costs: While this assumes zero fees, consider implementing a more
#    realistic fee structure to account for potential spread costs.
#
# 4. Optimization: Experiment with different parameters (e.g., lookback periods,
#    number of cryptocurrencies) to optimize the strategy's performance.
#
# 5. Robustness Testing: Perform out-of-sample testing and consider Monte Carlo
#    simulations to assess the strategy's robustness.
#
# 6. Market Regimes: Analyze the strategy's performance under different market
#    conditions and consider implementing regime-switching mechanisms.
#
# Remember, while backtests can provide valuable insights, they do not guarantee
# future performance. Always monitor your strategy closely and be prepared to
# make adjustments as market conditions change.