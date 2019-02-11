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
import numpy as np

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

oneticker = ['AAPL']

# Top 100 stocks analyzed.
real_tickers = ['SPY','AMZN','QQQ','AAPL','FB','NVDA','NFLX','IWM','BABA','EEM','MSFT','MU','GOOGL','TSLA','BAC','GOOG','BA','JPM','EFA','INTC','XLF','DIA','VXX','HYG','C','CSCO','IVV','XOM','FXI','XLE','TQQQ','WFC','GE','XLI','V','XLK','TWTR','WMT','AVGO','HD','TLT','CMCSA','CVX','GDX','BRK-B','T','CAT','JNJ','GLD','AMGN','XLU','UNH','DIS','AMAT','CRM','PFE','MA','MCD','BIDU','GS','VZ','SQ','LRCX','XLP','ORCL','ABBV','PG','IBM','XOP','VOO','ADBE','SPOT','LQD','AMD','MRK','XLV','IYR','QCOM','PYPL','IEMG','SMH','EWZ','XLY','UNP','TXN','LOW','NXPI','DWDP','UTX','MMM','VWO','CELG','EWJ','AABA','CVS','KO','LMT','MS','WYNN','PM']

# Start date set to January 1, 2016. End date is the current date.
start_date = '2016-01-01'
end_date = end

# Read data from online panel.
panel_data = data.DataReader(oneticker,'yahoo',start_date,end_date)

# Early functionality used to save data to a .csv file.
panel_data.to_csv('KeyosStockDataTickerValuesCSV.csv',sep=',')

print (panel_data)


#import data
data = pd.read_csv('data_stocks.csv')

# Drop date variable
data = data.drop(['DATE'], 1)

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a numpy array
data = data.values

# training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Import TensorFlow
import tensorflow as tf

# Define a and b as placeholders
a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)

# Define the addition
c = tf.add(a, b)

# Initialize the graph
graph = tf.Session()

# Run the graph
graph.run(c, feed_dict={a: 5, b: 4})


# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])


# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)



# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# Number of epochs and batch size
epochs = 10
batch_size = 256

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]


    print (len(y_train))
    print (batch_size)
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})


        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            #file_name = '/Users/yousif/Oasis/GitHub/CSI-4999/backend/' + str(e) + '_batch_' + str(i) + '.png'
            #file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'

            # or switch it back to jpg
            #plt.savefig(file_name)
            #plt.pause(0.01)


# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)
