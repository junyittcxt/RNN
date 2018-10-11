import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os, time, sys, sklearn
from sklearn.externals import joblib
from rnn_functions import *

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
# config = tf.ConfigProto(allow_soft_placement=True,
#                         log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#Parameters
NROWS = 300000
TARGET_TO_PREDICT = "USDCHF"

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 5

NUMLAYER = 2
NEURONS = 8
DROPOUT = 0.3
LEARNING_RATE = 0.0002
BATCH_SIZE = 256
# BATCH_SIZE = 1028
EPOCHS = 200
PATIENCE = 200
NAME = f"{TARGET_TO_PREDICT}-{SEQ_LEN}-{FUTURE_PERIOD_PREDICT}-{NUMLAYER}-{NEURONS}-{DROPOUT}-{LEARNING_RATE}-{BATCH_SIZE}-{EPOCHS}-{int(time.time())}"

PARAMS_INFO = dict(NROWS = NROWS, TARGET_TO_PREDICT = TARGET_TO_PREDICT, SEQ_LEN = SEQ_LEN, FUTURE_PERIOD_PREDICT = FUTURE_PERIOD_PREDICT,
                NUMLAYER = NUMLAYER, NEURONS = NEURONS, DROPOUT = DROPOUT, LEARNING_RATE = LEARNING_RATE, BATCH_SIZE = BATCH_SIZE,
                EPOCHS = EPOCHS, PATIENCE = PATIENCE,
                NAME = NAME
                )


#Load
# df = pd.read_csv("./DATA/x_82_ETF_FOREX_MIN_RETONLY.csv")
df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv", nrows = NROWS)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
print("Load: Done!")

#Pre-process (target)
df = create_target(df, TARGET_TO_PREDICT, FUTURE_PERIOD_PREDICT, cumulative_returns)
df = classify_target(df, "target", 0.0005, False)
print("Pre-Process: Done!")

#Split
prop = [0.5, 0.7, 0.85]
df_list_2 = split_df_by_prop(df, prop = prop)
print(len(df_list_2))
print([j.shape for j in df_list_2])

startdates = [j.index[0] for j in df_list_2]
enddates = [j.index[-1] for j in df_list_2]
print("startdates:", startdates)
print("enddates:", enddates)
PARAMS_INFO["startdates"] = startdates
PARAMS_INFO["enddates"] = enddates
print("Split: Done!")

#Scaling
scaler = sklearn.preprocessing.StandardScaler()
train_x, train_y, scaler, x_columns = preprocess_returns_df(df=df_list_2[0], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = True, same_prop = True, shuffle = True)
val_x, val_y, _, _ = preprocess_returns_df(df=df_list_2[1], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = True, shuffle = False)
# test_x, test_y, _, _ = preprocess_returns_df(df=df_list_2[2], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = True, shuffle = False)
run_dir = f'RUNS/{NAME}'
print("Scaling: Done!")

init_dir(run_dir)
joblib.dump(scaler, f'{run_dir}/rnn_scaler.pkl')
joblib.dump(x_columns, f'{run_dir}/x_columns.pkl')
joblib.dump(PARAMS_INFO, f'{run_dir}/PARAMS_INFO.pkl')
print("Meta-data Dump: Done!")

#Training
data = (train_x, train_y, val_x, val_y)
model, history = cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME, PATIENCE, logs_folder = TARGET_TO_PREDICT, models_folder=TARGET_TO_PREDICT, device_name = "/gpu:1")
print("Training: Done!")

# performance_binary(train_x, train_y, model, sample_type = "train")
# performance_binary(val_x, val_y, model, sample_type = "val")
# performance_binary(test_x, test_y, model, sample_type = "test")
