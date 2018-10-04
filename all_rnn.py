import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

import tensorflow as tf

import time
import sys
import sklearn
from sklearn.externals import joblib

from rnn_functions import *

#Show Device
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
# config = tf.ConfigProto(allow_soft_placement=True,
#                         log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#Parameters
NROWS = 200000
TARGET_TO_PREDICT = "SPY"

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 1

NUMLAYER = 5
NEURONS = 5
DROPOUT = 0.4
LEARNING_RATE = 0.0002
BATCH_SIZE = 32
# BATCH_SIZE = 1028
EPOCHS = 100
NAME = f"{TARGET_TO_PREDICT}-{SEQ_LEN}-{FUTURE_PERIOD_PREDICT}-{NUMLAYER}-{NEURONS}-{DROPOUT}-{LEARNING_RATE}-{BATCH_SIZE}-{EPOCHS}-{int(time.time())}"

PARAMS_INFO = dict(NROWS = NROWS, TARGET_TO_PREDICT = TARGET_TO_PREDICT, SEQ_LEN = SEQ_LEN, FUTURE_PERIOD_PREDICT = FUTURE_PERIOD_PREDICT,
                NUMLAYER = NUMLAYER, NEURONS = NEURONS, DROPOUT = DROPOUT, LEARNING_RATE = LEARNING_RATE, BATCH_SIZE = BATCH_SIZE,
                EPOCHS = EPOCHS, NAME = NAME
                )
#Load
# df = pd.read_csv("./DATA/x_82_ETF_FOREX_MIN_RETONLY.csv")
df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv", nrows = NROWS)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

#Pre-process
df = create_target(df, TARGET_TO_PREDICT, 3)
df = classify_target(df, "target")

#Split
prop = [0.5, 0.7, 0.85]
df_list_2 = split_df_by_prop(df, prop = prop)
print(len(df_list_2))
print([j.shape for j in df_list_2])

#Scaling
scaler = sklearn.preprocessing.StandardScaler()
train_x, train_y, scaler, x_columns = preprocess_returns_df(df=df_list_2[0], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = True, same_prop = True)
val_x, val_y, _, _ = preprocess_returns_df(df=df_list_2[1], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = True)
test_x, test_y, _, _ = preprocess_returns_df(df=df_list_2[2], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = True)
run_dir = f'RUNS/{NAME}'
init_dir(run_dir)
joblib.dump(scaler, f'{run_dir}/rnn_scaler.pkl')
joblib.dump(x_columns, f'{run_dir}/x_columns.pkl')
joblib.dump(PARAMS_INFO, f'{run_dir}/PARAMS_INFO.pkl')

#Training
data = (train_x, train_y, val_x, val_y)
model, history = cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME, logs_folder = TARGET_TO_PREDICT, models_folder=TARGET_TO_PREDICT, device_name = "/gpu:1")

print("========================")
performance_binary(train_x, train_y, model, sample_type = "test", threshold = 0.5, silence = False)
performance_binary(val_x, val_y, model, sample_type = "test", threshold = 0.5, silence = False)
performance_binary(test_x, test_y, model, sample_type = "test", threshold = 0.5, silence = False)


# performance_binary(train_x, train_y, model, sample_type = "train")
# performance_binary(val_x, val_y, model, sample_type = "val")
# performance_binary(test_x, test_y, model, sample_type = "test")
