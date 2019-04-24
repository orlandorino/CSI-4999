import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

from sklearn.preprocessing import MinMaxScaler

start = dt.datetime(2000, 1, 1)         # define the start date for fetching stock
end = dt.datetime.now()                 # define the end date for fetching stock

# create a pandas DataFrame for Apple Stock from Yahoo Finance from the defined start and end dates
df = web.DataReader("AAPL", 'yahoo', start, end)

df = df.reset_index()                   # resets index of df, makes date the first column
df = df.sort_values('Date')             # ensure that the data fetched is sorted correctly by date
df['Date'] = df['Date'].dt.date         # remove time stamp

# print (df.shape[0])                   # first dimension size
# print (df.shape[1])                   # second dimension size

ticks = 2 * int(df.shape[0] ** (1./2.))

# plot mid data for target stock
# plt.figure(figsize=(18,9))                                                        # figure size
# plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)                          # plot mid of data
# plt.xlabel('Date', fontsize=18)                                                   # x-label
# plt.ylabel('Mid Price', fontsize=19)                                              # y-label
# plt.xticks((range(0, df.shape[0], ticks)), df['Date'].loc[::ticks], rotation=45)  # x-label ticks
# plt.show()                                                                        # shows the plotted chart

high = df.loc[:,'High'].values          # stock high values
low = df.loc[:,'Low'].values            # stock low values
mid = (high + low)/2.0                  # derive stock mid values

n = mid.shape[0]
n = int(np.floor(0.8*n))          # index used for dividing training and testing data

train_data = mid[:n]            # divide train data
test_data = mid[n:]             # divide test data

scaler = MinMaxScaler()                     # scales all data to be in region of 0 to 1
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1,1)

smoothing_window_size = 3 * int(df.shape[0] / (df.shape[0] ** (1./3.)))

# train scaler with both training and smooth data
for di in range(0, n - smoothing_window_size, smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size:, :])
    train_data[di:di+smoothing_window_size, :] = scaler.transform(train_data[di:di+smoothing_window_size, :])

# normalize the remaining data
scaler.fit(train_data[di+smoothing_window_size:, :])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# reshape train and test data, then normalize test data
train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)


EMA = 0.0
gamma = 0.1

# smooth data with exponential moving average
for ti in range(n):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# visualization and testing
all_mid_data = np.concatenate([train_data,test_data],axis=0)

window_size = int(df.shape[0] * .01)
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
# plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
# plt.xticks((range(0, df.shape[0], ticks)), df['Date'].loc[::ticks], rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend(fontsize=18)
# plt.show()

N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
# plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
# plt.xticks((range(0, df.shape[0], ticks)), df['Date'].loc[::ticks], rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend(fontsize=18)
# plt.show();

# ========================= Data Generation Class =====================================

class DataGeneratorSeq(object):

    def __init__(self,prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels

    def unroll_batches(self):
        unroll_data,unroll_labels = [],[]
        init_data, init_label = None, None
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))

# ====================================================================================


dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
    print('Unrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ',dat )
    print('\tOutput: ',lbl)


D = 1                               # Dimension of Data = 1 D
num_unrollings = 50                 # # of time steps into the future
batch_size = 500                    # # of samples in batch                         # may need to change this
num_nodes = [200,200,150]           # # of hidden nodes in each layer of LSTM
n_layers = len(num_nodes)           # # of layers
dropout = 0.2                       # dropout value

tf.reset_default_graph()            # resets because script is run multiple times

train_inputs, train_outputs = [],[]         # input data

# unroll input over time defined by placeholders for each time step
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, D],name='train_inputs_%d' % ui))
    train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1], name='train_outputs_%d' % ui))

# define LSTM cell
lstm_cells = [
    tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
                            state_is_tuple=True,
                            initializer=tf.contrib.layers.xavier_initializer()) for li in range(n_layers)]

# define drop of LSTM cell
drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0,
                                                 output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout)
                   for lstm in lstm_cells]

drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

# variables for the LSTM's three layers and Linear Regression
w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))


c, h = [],[]                # cell state and hidden state
initial_state = []          # initial state

for li in range(n_layers):
  c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
  h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
  initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

# concatenates the data and transforms it into specific format
all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)

# all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state), time_major=True, dtype=tf.float32)

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])
all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)
split_outputs = tf.split(all_outputs,num_unrollings, axis=0)

print ('Defining Training Loss')
loss = 0.0

with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)] +
                             [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(0.5 * (split_outputs[ui]-train_outputs[ui]) ** 2)

print ('Learning Rate Decay Operations')

global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.assign(global_step, global_step + 1)
tf_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
tf_min_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

learning_rate = tf.maximum(tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5,
                                                      staircase=True), tf_min_learning_rate)

print ('TF Optimization Operations')

optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(zip(gradients, v))


print('Defining Prediction Related TF Functions')

sample_inputs = tf.placeholder(tf.float32, shape=[1, D])

# maintaining LSTM state for prediction stage
sample_c, sample_h, initial_sample_state = [],[],[]
for li in range(n_layers):
  sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
  sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
  initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))

reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                               *[tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0),
                                                 initial_state=tuple(initial_sample_state),
                                                 time_major=True,
                                                 dtype=tf.float32)

with tf.control_dependencies([tf.assign(sample_c[li], sample_state[li][0]) for li in range(n_layers)] +
                             [tf.assign(sample_h[li], sample_state[li][1]) for li in range(n_layers)]):
    sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)

print('\tAll done')

epochs = 10                 # change back to 30
valid_summary = 1           # interval for test predictions
n_predict_once = 50         # # of steps predicted for

train_seq_length = train_data.size          # training data length
train_mse_ot = []                           # Accumulate Train Loss
test_mse_ot = []                            # Accumulate Test Loss
predictions_over_time = []                  # Accumulate Predictions

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

loss_nondecrease_count = 0              # decaying learning rate
loss_nondecrease_threshold = 2          # if error hasn't increased in this amount of steps, decrease learning rate

print('Initialized')

average_loss = 0

data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)           # define data generation
x_axis_seq = []


train_data = mid[:n]            # divide train data
test_data = mid[n:]             # divide test data

scaler = MinMaxScaler()                     # scales all data to be in region of 0 to 1
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1,1)

smoothing_window_size = 3 * int(df.shape[0] / (df.shape[0] ** (1./3.)))

print ("smoothing window ")
print (smoothing_window_size)

# Points you start your test predictions from # figure this 50 out
test_points_seq = np.arange(n, mid.shape[0], 50).tolist()            # pointers to test predictions

for ep in range(epochs):

    # ========================= Training =====================================

    for step in range(train_seq_length // batch_size):

        u_data, u_labels = data_gen.unroll_batches()

        feed_dict = {}
        for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
            feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
            feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

        feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

    # ============================ Validation ==============================
    if (ep+1) % valid_summary == 0:

        average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))

        # average loss
        if (ep+1)%valid_summary==0:
            print('Average loss at step %d: %f' % (ep+1, average_loss))

        train_mse_ot.append(average_loss)

        average_loss = 0                        # reset loss

        predictions_seq = []

        mse_test_loss_seq = []

        # ===================== Updating State and Making Predictions =======================
        for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []

            if (ep+1)-valid_summary == 0:
                x_axis = []             # calculate x axis values for first validation epoch

            # feed past behavior of stock prices to prediction for points onwards
            for tr_i in range(w_i-num_unrollings+1, w_i-1):
                current_price = all_mid_data[tr_i]
                feed_dict[sample_inputs] = np.array(current_price).reshape(1, 1)
                _ = session.run(sample_prediction,feed_dict=feed_dict)

                feed_dict = {}

                current_price = all_mid_data[w_i - 1]

                feed_dict[sample_inputs] = np.array(current_price).reshape(1, 1)

                # make predictions for this many steps, each one using previous prediction
                for pred_i in range(n_predict_once):

                    pred = session.run(sample_prediction, feed_dict=feed_dict)

                    our_predictions.append(np.asscalar(pred))

                    feed_dict[sample_inputs] = np.asarray(pred).reshape(-1, 1)

                    if (ep + 1) - valid_summary == 0:
                        x_axis.append(w_i + pred_i)         # only calculate x axis values for first validation epoch

                # mse_test_loss += 0.5*(pred - all_mid_data[w_i + pred_i])**2

                session.run(reset_sample_states)

                predictions_seq.append(np.array(our_predictions))

                mse_test_loss /= n_predict_once
                mse_test_loss_seq.append(mse_test_loss)

                if (ep + 1) - valid_summary == 0:
                    x_axis_seq.append(x_axis)

            current_test_mse = np.mean(mse_test_loss_seq)

            # learning rate decay logic
            if len(test_mse_ot) > 0 and current_test_mse > min(test_mse_ot):
                loss_nondecrease_count += 1
            else:
                loss_nondecrease_count = 0


            if loss_nondecrease_count > loss_nondecrease_threshold:
                session.run(inc_gstep)
                loss_nondecrease_count = 0
                print('\tDecreasing learning rate by 0.5')

            test_mse_ot.append(current_test_mse)
            print('\tTest MSE: %.5f' % np.mean(mse_test_loss_seq))
            predictions_over_time.append(predictions_seq)
            print('\tFinished Predictions')


best_prediction_epoch = 8 # replace this with the epoch that you got the best results when running the plotting code
# switch back to 28

print (x_axis_seq)
print (predictions_over_time)
print (len(predictions_over_time))

plt.figure(figsize = (18, 18))
plt.subplot(2, 1, 1)
plt.plot(range(df.shape[0]), all_mid_data, color='b')

# Plotting how the predictions change over time
# Plot older predictions with low alpha and newer predictions with high alpha

start_alpha = 0.25
alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))

#for p_i, p in enumerate(predictions_over_time[::3]):
#    for xval, yval in zip(x_axis_seq, p):
#        plt.plot(xval,yval,color='r',alpha=alpha[p_i])


plt.title('Evolution of Test Predictions Over Time',fontsize=18)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.xlim(11000,12500)

plt.subplot(2, 1, 2)

# Predicting the best test prediction you got
# plt.plot(range(df.shape[0]),all_mid_data,color='b')
# for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
#     plt.plot(xval,yval,color='r')
#
# plt.title('Best Test Predictions Over Time',fontsize=18)
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.xlim(11000,12500)
# plt.show()