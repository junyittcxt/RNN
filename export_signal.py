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
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

SKIPROWS = np.arange(1,300000,1)
SKIPROWS = None
NROWS = 300000

#Load
df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv", header=0, skiprows = SKIPROWS, nrows = NROWS)
# df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
print("Load: Done!")

#Import scaler
run_dir = "RUNS/EURUSD-60-2-2-10-0.3-0.0002-256-200-1538974780"
run_dir = "RUNS/EURUSD-60-5-2-5-0.3-0.0002-256-200-1538993858"
run_dir = "RUNS/USDCHF-60-5-2-5-0.3-0.0002-256-200-1538996078"

scaler = joblib.load(f'{run_dir}/rnn_scaler.pkl')
x_columns = joblib.load(f'{run_dir}/x_columns.pkl')
PARAMS_INFO = joblib.load(f'{run_dir}/PARAMS_INFO.pkl')
print("Import Scaler: Done!")

df[x_columns] = scaler.transform(df[x_columns].values)
tdf = df[x_columns]
SEQ_LEN = PARAMS_INFO["SEQ_LEN"]

sequential_data = []
prev_days = deque(maxlen = SEQ_LEN)

sequential_list_dict = []
for i, j  in zip(tdf.values, tdf.index):
    prev_days.append(i)
    if len(prev_days) == SEQ_LEN:
        sequential_list_dict.append(dict(t = j, x = np.array(prev_days)))

print("Transform: Done!")


model_input = "./tohost_model/RNN_Final-031-0.525.model"
model_input = "./tohost_model/RNN_Final-071-0.6957-0.5541.model"
model_input = "./tohost_model/RNN_Final-086-0.6787-0.5830.model"


rnn_model = tf.keras.models.load_model(filepath = model_input, custom_objects=None, compile=False)

print("Model Loaded!")


x_list_3d = []
t_list = []

for seq_dict in sequential_list_dict:
    t = seq_dict["t"]
    x = seq_dict["x"]
    t_list.append(t)
    x_list_3d.append(x)

x_list_3d = np.array(x_list_3d)

print("Dimension: ", x_list_3d.shape)


y = rnn_model.predict(np.array(x_list_3d))

print("Predict: Done!")
pred_df = pd.DataFrame(dict(Date = t_list, signal_raw = y.reshape((-1))))
print(pred_df.head(5))
print(pred_df.tail(5))

instrument_name = PARAMS_INFO["TARGET_TO_PREDICT"]
pred_df.to_csv("rnn_signal_{}.csv".format(instrument_name), index = False)
print("Export: Done!")
