# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:45:07 2018

@author: Workstation
"""

import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import sys


sys.path.append("C:/Users/Workstation/Desktop/tf_ml/Step3_ML1_Model")
import MLPerformance_Class as mlp

SEQ_LEN = 150
FUTURE_PERIOD_PREDICT = 5
RATIO_TO_PREDICT = "BTC-USD"
LEARNING_RATE = 0.0005
EPOCHS = 100
BATCH_SIZE = 128
NEURONS = 6
NAME = f"{RATIO_TO_PREDICT}-{NEURONS}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df, same_prop = False):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()

    df.dropna(inplace = True)

    for col in df.columns:
        if col != "target":
            df[col] = preprocessing.scale(df[col].values)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)
#    print(df.head())

    timestamp = []
    for i,j in zip(df.values, df.index):
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            timestamp.append(j)

    random.shuffle(sequential_data)


#    random.shuffle(buys)
#    random.shuffle(sells)

    if same_prop:
        buys = []
        sells = []

        for seq, target in sequential_data:
            if target == 0:
                sells.append([seq, target])
            elif target == 1:
                buys.append([seq, target])

        lower = min(len(buys), len(sells))
        buys = buys[:lower]
        sells = sells[:lower]
        sequential_data = buys + sells

#    random.shuffle(sequential_data)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

#    return np.array(X), np.array(y), df
    return np.array(X), np.array(y), np.array(timestamp), df

scaler = preprocessing.StandardScaler()

colnames = ["time", "low", "high", "open", "close", "volume"]
df = pd.read_csv("LTC-USD.csv", names = colnames)

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

main_df = pd.DataFrame()

for ratio in ratios:
    dataset = f"{ratio}.csv"
    df = pd.read_csv(dataset, names = colnames)
    df.rename(columns = {"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace = True)

    df.set_index("time", inplace = True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method = "ffill", inplace = True)
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
main_df.dropna(inplace = True)

times = sorted(main_df.index.values)
last_5pct = times[-int(0.10*len(times))]
last_25pct = times[-int(0.25*len(times))]

training_main_df = main_df[(main_df.index < last_25pct)]
validation_main_df = main_df[(main_df.index >= last_25pct) & (main_df.index < last_5pct)]
test_main_df = main_df[(main_df.index >= last_5pct)]
#test_main_df.shape
#test_y.shape

train_x, train_y, train_t, train_df = preprocess_df(training_main_df, True)