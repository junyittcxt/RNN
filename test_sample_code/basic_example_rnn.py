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

random.seed(111)
rng = pd.date_range(start='2000', periods=210, freq='M')
ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
# ts.plot(c='b', title='Sample Time Series')
# plt.show()

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



tsa = np.array(ts)
num_periods = 20
f_horizon = 1

x_data = tsa[:(len(tsa)-(len(tsa) % num_periods))]
x_batches = x_data.reshape(-1, 20, 1)

y_data = tsa[1:(len(tsa) - (len(tsa) % num_periods)) + f_horizon]
y_batches = y_data.reshape(-1, 20, 1)

def split_data(series, forecast, num_periods):
    test_x_setup = tsa[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
    testY = tsa[-(num_periods):].reshape(-1, 20, 1)
    return testX, testY

X_test, Y_test = split_data(tsa, f_horizon, num_periods)
Y_test.shape
tf.reset_default_graph()

inputs = 1
hidden = 200
output = 1
learning_rate = 0.001

np.array(x_batches).shape
np.array(y_batches).shape

x = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden, activation = tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

import time
epochs = 2000
t0 = time.time()
with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict = {x : x_batches, y : y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict = {x : x_batches, y : y_batches})
            print(ep, "\tMSE: ", mse)
    y_pred = sess.run(outputs, feed_dict = {x : X_test})

t1 = time.time()
print("Time taken: ", str(t1-t0))

plt.close()
plt.title("Forecast vs. Actual", fontsize=12)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label = "Actual")
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label = "Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")

plt.show()
