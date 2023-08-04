# Monte-Carlo-Simulation-Risk-Management
The Monte Carlo simulation is a statistical method that uses random sampling to generate a range of possible outcomes in a model. In finance, Monte Carlo simulations are often used to estimate the probability of different investment outcomes and to assess the risk associated with different investment strategies.

In this specific model, the Monte Carlo simulation is used to estimate the potential returns of a stock portfolio over a given time period. The model uses historical stock data to calculate the mean returns and covariance matrix of the stocks in the portfolio. It then generates a set of random weights for the portfolio and calculates the returns of the portfolio based on these weights. The Monte Carlo simulation is used to generate a range of possible returns for the portfolio by simulating the daily returns of each stock in the portfolio over the time period. The simulation uses a normal distribution to generate random numbers, which are then multiplied by the Cholesky decomposition of the covariance matrix to generate the daily returns of each stock.

*To generate the correlated random numbers, the simulation starts with a set of uncorrelated random numbers that are normally distributed. These numbers are generated using the NumPy function np.random.normal(). The mean of the normal distribution is set to zero and the standard deviation is set to one. 
The covariance matrix of the stock returns is then calculated from the historical stock data. The covariance matrix describes the relationships between the returns of the different stocks in the portfolio. For example, if two stocks tend to move in the same direction, their returns will be positively correlated, and if they tend to move in opposite directions, their returns will be negatively correlated.
The Cholesky decomposition is then applied to the covariance matrix to obtain a lower triangular matrix. This lower triangular matrix is then used to transform the set of uncorrelated random numbers into a set of correlated random numbers, obtained by multiplying the lower triangular matrix by the set of uncorrelated random numbers.
The resulting set of correlated random numbers is then used to generate the daily returns for each stock in the portfolio. The daily returns are calculated by adding the mean daily returns of each stock, as well as the product of the Cholesky factors and the set of correlated random numbers, weighted by the standard deviation of the stock returns.
Overall, the use of the Cholesky decomposition allows the Monte Carlo simulation to generate realistic and correlated random numbers for the daily returns of each stock in the portfolio, which in turn allows for a more accurate estimation of the potential returns of the portfolio.*

The simulation is run multiple times to generate a range of possible outcomes for the portfolio. The final portfolio value for each simulation is calculated by multiplying the initial portfolio value by the cumulative product of the daily returns.

**The VaR (Value at Risk) and CVaR (Conditional Value at Risk)**, which are calculated based on the Monte Carlo simulation results, can provide valuable information for evaluating the risk associated with the portfolio and establishing risk management strategies.

* The **VaR** represents the maximum potential loss that the portfolio may suffer over a given time period at a given confidence level. In this model, the VaR is calculated as the difference between the initial value of the portfolio and the 5th percentile of the distribution of final portfolio values generated by the Monte Carlo simulation. This means that there is a 5% chance that the portfolio will experience a loss equal to or greater than the VaR over the time period.

* The **CVaR**, also known as Expected Shortfall, is a measure of the expected loss of the portfolio beyond the VaR. In other words, the CVaR represents the average loss that the portfolio is expected to incur when the portfolio returns fall below the VaR. In this model, the CVaR is calculated as the difference between the initial value of the portfolio and the average of the returns that fall below the VaR.

The values that come out of the model for the selected stocks are VaR $1276.82 and CVaR $1525.65. These values indicate that there is a 5% probability that the portfolio may suffer a loss of at least $1276.82 over the given time period. Moreover, in the event that the portfolio returns fall below the VaR, the average loss is expected to be $1525.65.
