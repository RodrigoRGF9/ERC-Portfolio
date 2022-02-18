# Daemon Intern Test II - Rodrigo Gesuatto - 18/02/2022

import statistics as st
import numpy as np
import yfinance as yf

# DATA MANIPULATION

#get the historical prices for each ticker
tickers = "PETR4.SA VALE3.SA SUZB3.SA CYRE3.SA GGBR4.SA RENT3.SA NTCO3.SA RAIL3.SA MRVE3.SA TEND3.SA"
startDate = "2021-8-1"
endDate = "2022-12-1"
period = "1d"
data = yf.download(tickers,start=startDate,end=endDate,period=period)

#get Close column
closes = data['Close']
#get returns
returns = closes.pct_change(1)  #get daily return for the stocks  
returns = returns.iloc[1:,:]    #remove the first row

tickers_list = tickers.split(sep=" ") # list of tickers
stocks_list = {i: tickers_list[i] for i in range(len(tickers_list))} #dict to reference stock 'i' and its ticker

stocks_returns = [] # initializing list of daily returns' lists

for ticker in tickers_list:
    stocks_returns.append(returns[ticker].values.tolist())

start_21days_range = 0
end_21days_range = 21
stocks_returns = np.array(stocks_returns) # total sample
stocks_returns_21days = stocks_returns[:,start_21days_range:end_21days_range] #slicing to get 21 days 'sub-sample'

# SAMPLE DESCRIPTION

# The total sample obtained after manipulation consists of daily returns (in an period sufficient to analyze three
# month performance) of 10 stocks. The data is stored in an list of daily returns lists (one for each stock)
# The datatype of each daily return is float
# For the determination of variance-covariance matrix, an auxiliar sub-matrix was created, containing a sample
# of 21 days of daily returns for each stock. This matrix is updated at the end of each day  

# ALLOCATION AND REBALANCING

# Function just to print the rebalance after a period
def printBalance(interval,period,w,stocks_list, mth_acum_return,acum_return,risk_P):
    rf = 0.075 # BZ risk free rate
    sharpe = (acum_return-rf)/risk_P 
    print(f"After {interval} {period} the portfolio's rebalance is:")
    for i,weight in enumerate(w):
        print(f" |  The weight of stock {stocks_list[i]} must be rebalanced by: {weight*100} %")
    print("\n------ Portfolio Metrics ------\n") 
    print(f"The portfolio's return this {period} was {mth_acum_return*100} %")
    print(f"The portolio's return until now is {acum_return*100} %")
    print(f"The portfolio's current volatility is {risk_P*100} %")
    print(f"Resulting a sharpe ratio of {sharpe} %\n")


n_stocks = len(tickers_list)

varCov = np.cov(stocks_returns_21days) # variance-covariance matrix

rebalanceTime = 21
day = rebalanceTime + 1

# Initialization of weights' array
#   Here is important to mention that I assumed a equal weight for all stocks
#   since I guess the most important are the weight's deltas after assembling.
#   So, the assembled portfolio is not ERC, but it would be if I rebalanced
#   the weight after the first day active

w0 = np.array([0.1]*n_stocks) 
w = np.array([0.1]*n_stocks)
deltaW = [0]*n_stocks


acumulated_portfolio_return = 0
monthly_portfolio_return = 0
month = 1

# Loop that evaluate the new weights daily for each stock and
# print the rebalances needed to the portfolio become ERC
# after 1 day, 1 month, 2 months and 3 months.

while month != 3: 

    daily_portfolio_return = np.matmul(w,stocks_returns[:,day-1])
    monthly_portfolio_return += daily_portfolio_return

    risk_P = np.matmul(np.matmul(w,varCov),w.T)**0.5

    for i in range(len(stocks_returns)): # this 'for' will calculate the new weight each stock will have in the portfolio
                                         # based on their daily return (conttibution to portfolio's risk increasing/decreasing)

        sigma_ix = np.cov([stocks_returns_21days[i],np.matmul(w,stocks_returns_21days)])[1,0]

        stock_beta = sigma_ix/(risk_P**2)

        w[i] = (stock_beta**-1)/n_stocks

        deltaW[i] = w[i]-w0[i]    

    if day == rebalanceTime + 1: 
        printBalance(1,'day', deltaW, stocks_list,monthly_portfolio_return,monthly_portfolio_return,risk_P)
    if day%rebalanceTime == 0:
        month += 1
        acumulated_portfolio_return += monthly_portfolio_return
        printBalance(month,'month(s)', deltaW, stocks_list,monthly_portfolio_return,acumulated_portfolio_return,risk_P)
        monthly_portfolio_return = 0


    day += 1
    start_21days_range += 1
    end_21days_range += 1
    stocks_returns_21days = stocks_returns[:,start_21days_range:end_21days_range]
    varCov = np.cov(stocks_returns_21days)

print("\nREBALANCE CONCLUDED")
