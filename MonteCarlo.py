import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm, t
import matplotlib.pyplot as plt

# Import data
def getData(stocks, start, end):
    # Download stock data from Yahoo Finance
    stockData = yf.download(stocks, start=start, end=end)['Adj Close']
    # Calculate daily returns, mean returns, and covariance matrix
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# Portfolio Performance
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    # Calculate portfolio returns and standard deviation
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(Time)
    return returns, std

# Define the list of stocks in the portfolio and the start and end dates for the data
stockList = ['CBA.AX', 'BHP.AX', 'TLS.AX', 'NAB.AX', 'WBC.AX', 'STO.AX']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)

# Get the stock data, mean returns, and covariance matrix
returns, meanReturns, covMatrix = getData(stockList, start=startDate, end=endDate)
returns = returns.dropna()

# Generate random weights for the portfolio and normalize them
weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

# Calculate the returns of the portfolio
returns['portfolio'] = returns.dot(weights)

# Monte Carlo Method
mc_sims = 400 # number of simulations
T = 100 #timeframe in days

# Create a matrix of mean returns for each stock
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

# Create a matrix to store the simulated portfolio values
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

# Set the initial portfolio value
initialPortfolio = 10000

# Loop through the Monte Carlo simulations
for m in range(0, mc_sims):
    # Generate random numbers from a normal distribution
    Z = np.random.normal(size=(T, len(weights)))
    # Use Cholesky decomposition to generate daily returns
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    # Calculate the total portfolio value for each simulation
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

# Plot the simulated portfolio values over time
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")

# Calculate the final portfolio values for each simulation
portResults = pd.Series(portfolio_sims[-1,:])

# Calculate the VaR and CVaR of the portfolio based on the simulation results
VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

# Print the VaR and CVaR of the portfolio
print('VaR ${}'.format(round(VaR,2)))
print('CVaR ${}'.format(round(CVaR,2)))
