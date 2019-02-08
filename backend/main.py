"""
I think we can get rid of this comment, but saving just in case.

<<<<<<< HEAD
import pandas_datareader as pdr
import datetime
aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2016, 1, 1),
                          end=datetime.datetime(2019, 1, 1))
=======

import numpy as np


>>>>>>> 878dc3770e2651e55280ab1e8d4bf310306823ca
"""


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import data
from pandas import ExcelWriter

style.use('ggplot')

# Keyos Stock Analyzer Algorithm

# Stock information is recorded from the date listed as "start".  January 16th chosen to allow 3+ years of information.
# We want this information to adapt to everyday situations, thus "end" allows the data to stay current.
# DataReader is a Pandas function which allows us to extract data from various internet sources.

# Sprint 3 bug/error: January 1, 2016 through January 3, 2016 not appearing in output data set.
# Output .csv and interpreter show dates missing as well.
# Earlier dates in December of 2015 are shown without error.
# Data from those dates may have been unrecorded; other problems may have occurred.
# Error is non critical. Sample size is nearly unaffected.


start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()
df = web.DataReader("AAPL", 'yahoo', start, end)

# Print the collected information.
print (df)

# 1 Ticker corresponds to 1 stock. Each has a string of characters uniquely defining each one.
# 3 stock tickers chosen randomly for simple queries or debugging.
tickers = ['AAPL','MSFT','TSLA']

# Top 100 stocks analyzed.
real_tickers = ['SPY','AMZN','QQQ','AAPL','FB','NVDA','NFLX','IWM','BABA','EEM','MSFT','MU','GOOGL','TSLA','BAC','GOOG','BA','JPM','EFA','INTC','XLF','DIA','VXX','HYG','C','CSCO','IVV','XOM','FXI','XLE','TQQQ','WFC','GE','XLI','V','XLK','TWTR','WMT','AVGO','HD','TLT','CMCSA','CVX','GDX','BRK-B','T','CAT','JNJ','GLD','AMGN','XLU','UNH','DIS','AMAT','CRM','PFE','MA','MCD','BIDU','GS','VZ','SQ','LRCX','XLP','ORCL','ABBV','PG','IBM','XOP','VOO','ADBE','SPOT','LQD','AMD','MRK','XLV','IYR','QCOM','PYPL','IEMG','SMH','EWZ','XLY','UNP','TXN','LOW','NXPI','DWDP','UTX','MMM','VWO','CELG','EWJ','AABA','CVS','KO','LMT','MS','WYNN','PM']

# Start date set to January 1, 2016. End date is the current date.
start_date = '2016-01-01'
end_date = end

# Read data from online panel.
panel_data = data.DataReader(real_tickers,'yahoo',start_date,end_date)

# Early functionality used to save data to a .csv file.
panel_data.to_csv('KeyosStockDataTickerValuesCSV.csv',sep=',')

# print (panel_data)

