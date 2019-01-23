import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import data
from pandas import ExcelWriter


# style.use('ggplot')

# start = dt.datetime(2019, 1, 1)
# end = dt.datetime.now()
# df = web.DataReader("AAPL", 'yahoo', start, end)

# print (df)


# tickers we use for the sake of testing
tickers = ['AAPL','MSFT','TSLA']

# stocks that we plan to analyze, top 100 most popular stocks
real_tickers = ['SPY','AMZN','QQQ','AAPL','FB','NVDA','NFLX','IWM','BABA','EEM','MSFT','MU','GOOGL','TSLA','BAC','GOOG','BA','JPM','EFA','INTC','XLF','DIA','VXX','HYG','C','CSCO','IVV','XOM','FXI','XLE','TQQQ','WFC','GE','XLI','V','XLK','TWTR','WMT','AVGO','HD','TLT','CMCSA','CVX','GDX','BRK-B','T','CAT','JNJ','GLD','AMGN','XLU','UNH','DIS','AMAT','CRM','PFE','MA','MCD','BIDU','GS','VZ','SQ','LRCX','XLP','ORCL','ABBV','PG','IBM','XOP','VOO','ADBE','SPOT','LQD','AMD','MRK','XLV','IYR','QCOM','PYPL','IEMG','SMH','EWZ','XLY','UNP','TXN','LOW','NXPI','DWDP','UTX','MMM','VWO','CELG','EWJ','AABA','CVS','KO','LMT','MS','WYNN','PM']

start_date = '2019-01-01'
end_date = '2019-01-20'

panel_data = data.DataReader(real_tickers,'yahoo',start_date,end_date)

# export to csv
panel_data.to_csv('PythonExportCSV.csv',sep=',')

# print (panel_data)
