# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 03:45:58 2018

@author: Workstation
"""

import tensorflow as tf
import pandas as pd
import numpy as np

import os
import matplotlib
import matplotlib.pyplot as plt
import random
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as tfmetrics
import tensorflow.contrib.rnn as tfrnn

#random.seed(111)
#rng = pd.date_range(start='2000', periods=210, freq='M')
#ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
#ts.plot(c='b', title='Sample Time Series')
#plt.show()
#
#tsa = np.array(ts)
#num_periods = 20
#f_horizon = 1
#
#x_data = tsa[:(len(tsa)-(len(tsa) % num_periods))]
#x_batches = x_data.reshape(-1, 20, 1)
#
#y_data = tsa[1:(len(tsa) - (len(tsa) % num_periods)) + f_horizon]
#y_batches = y_data.reshape(-1, 20, 1)

#def split_data(series, forecast, num_periods):
#    test_x_setup = tsa[-(num_periods + forecast):]
#    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
#    testY = tsa[-(num_periods):].reshape(-1, 20, 1)
#    return testX, testY
#
#X_test, Y_test = split_data(tsa, f_horizon, num_periods)

tf.reset_default_graph()

#inputs = 1
hidden = 200
#output = 1
learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, shape_x[1], shape_x[2]])
y = tf.placeholder(tf.float32, [None, 1])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden, activation = tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

val = tf.transpose(rnn_output, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
weight = tf.Variable(tf.truncated_normal([hidden, shape_x[2]]))
bias = tf.Variable(tf.constant(0.1, shape=[shape_x[2]]))
prediction = tf.matmul(last, weight) + bias
 
#stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
#stacked_outputs = tf.layers.dense(stacked_rnn_output, 1)
#outputs = tf.reshape(stacked_outputs, [-1, 1])
#
loss = tf.reduce_sum(tf.square(prediction - y))
#
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 2000

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        print(ep)
        sess.run(training_op, feed_dict = {x : train_x, y : train_y})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict = {x : train_x, y : train_y})
            print(ep, "\tMSE: ", mse)
    y_pred = sess.run(prediction, feed_dict = {x : val_x})
#    y_pred = sess.run(outputs, feed_dict = {x : val_x})
    
lstm_graph = tf.Graph()
with lstm_graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size])
    targets = tf.placeholder(tf.float32, [None, config.input_size])
    learning_rate = tf.placeholder(tf.float32, None)

def _create_one_cell():
    return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
    if config.keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        
        
plt.close()
plt.title("Forecast vs. Actual", fontsize=12)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label = "Actual")
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label = "Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")

plt.show()# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 04:38:41 2018

@author: Workstation
"""

