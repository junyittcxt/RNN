import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

import tensorflow as tf

import time
import sys
import sklearn
from sklearn.externals import joblib

# sys.path.append("C:/Users/Workstation/Desktop/tf_ml/Step3_ML1_Model")
# sys.path.append("/home/workstation/Desktop/tf_ml/Step3_ML1_Model")
# import MLPerformance_Class as mlp

from rnn_functions import *

#Show Device
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


RATIO_TO_PREDICT = "BTC-USD"

SEQ_LEN = 150
FUTURE_PERIOD_PREDICT = 5

NUMLAYER = 1
NEURONS = 10
DROPOUT = 0.4
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
EPOCHS = 100
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-{FUTURE_PERIOD_PREDICT}-{NUMLAYER}-{NEURONS}-{DROPOUT}-{LEARNING_RATE}-{BATCH_SIZE}-{EPOCHS}-{int(time.time())}"

#Load
main_df = sample_cc_main_df(RATIO_TO_PREDICT, FUTURE_PERIOD_PREDICT)

#Split
training_main_df, validation_main_df, test_main_df = split_df(main_df, [0.1,0.25])

#Scaling
scaler = sklearn.preprocessing.StandardScaler()
train_x, train_y, train_t, train_df, scaler, x_columns = preprocess_df(df=training_main_df, scaler = scaler, SEQ_LEN = SEQ_LEN, fit = True, same_prop = False)
val_x, val_y, val_t, val_df, _, _ = preprocess_df(df = validation_main_df, scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = False)
test_x, test_y, test_t, test_df, _, _ = preprocess_df(df = test_main_df, scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = False)
joblib.dump(scaler, "./rnn_scaler.pkl")
joblib.dump(x_columns, "./x_columns.pkl")

#Training
data = (train_x, train_y, val_x, val_y)
model, history = cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME)



#

#
# def performance(x, y, model, sample = "sample"):
#     y_actual = np.array(y).flatten()
#     pred_yy = np.array(model.predict(x))
#     y_pred = pred_yy[:,1]
#     print(mlp.performance_binary(y_actual, y_pred, sample_type = sample))
#
# performance(train_x, train_y, model, sample = "train")
# performance(val_x, val_y, model, sample = "val")
# performance(test_x, test_y, model, sample = "test")
