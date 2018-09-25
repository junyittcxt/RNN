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
import sklearn
from sklearn.externals import joblib 


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


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

def preprocess_df(df, scaler, fit = True, same_prop = False):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()

    df.dropna(inplace = True)

    x_columns = [c for c in df.columns if c != "target"]
    if fit:
        scaler.fit(df[x_columns].values)
        
    df[x_columns] = scaler.transform(df[x_columns].values)

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
    return np.array(X), np.array(y), np.array(timestamp), df, scaler, x_columns


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


scaler = sklearn.preprocessing.StandardScaler()
train_x, train_y, train_t, train_df, scaler, x_columns = preprocess_df(df=training_main_df, scaler = scaler, fit = True, same_prop = False)
val_x, val_y, val_t, val_df, _, _ = preprocess_df(df = validation_main_df, scaler = scaler, fit = False, same_prop = False)
test_x, test_y, test_t, test_df, _, _ = preprocess_df(df = test_main_df, scaler = scaler, fit = False, same_prop = False)
joblib.dump(scaler, "C:/Users/Workstation/Desktop/RNN/rnn_scaler.pkl")
joblib.dump(x_columns, "C:/Users/Workstation/Desktop/RNN/x_columns.pkl")



#os.environ["CUDA_VISIBLE_DEVICES"]="1"

tf.reset_default_graph()

model = Sequential()
model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.1))
model.add(BatchNormalization())


model.add(Dense(np.round(NEURONS/2,0), activation = "relu"))
# model.add(Dropout(0.1))

model.add(Dense(2, activation = "softmax"))

opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = 1e-6)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
              )

#model.compile(loss='binary_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy']
#              )

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = tf.keras.callbacks.ModelCheckpoint("models/{}.model".format(filepath),
                                                   monitor='val_acc',
                                                   verbose=1,
                                                   save_best_only=True, save_weights_only=False,
                                                   mode='max')

history = model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_x, val_y),
                    callbacks=[tensorboard, checkpoint])


def performance(x, y, model, sample = "sample"):
    y_actual = np.array(y).flatten()
    pred_yy = np.array(model.predict(x))
    y_pred = pred_yy[:,1]
    print(mlp.performance_binary(y_actual, y_pred, sample_type = sample))

performance(train_x, train_y, model, sample = "train")
performance(val_x, val_y, model, sample = "val")
performance(test_x, test_y, model, sample = "test")


# import tensorflow as tf
# filepath = "C:/Users/Workstation/Desktop/rnn_crypto/models/RNN_Final-26-0.571.model"
# m2 = tf.keras.models.load_model(
#     filepath = filepath,
#     custom_objects=None,
#     compile=False
# )
#
#
# performance(train_x, train_y, m2, sample = "train")
# performance(val_x, val_y, m2, sample = "val")
# performance(test_x, test_y, m2, sample = "test")

# pred_y = m2.predict(test_x)[:,1]
#
#
# test_main_df["target_pct"] = test_main_df["future"].pct_change()
# test_df_2 = test_main_df.ix[test_t]
# test_df_2['pred'] = pred_y >= 0.5


# from datetime import datetime
# datetime.fromtimestamp(1528968660).strftime("%A, %B %d, %Y %I:%M:%S")
# datetime.fromtimestamp(1535215260).strftime("%A, %B %d, %Y %I:%M:%S")
#
# commission = 0.002
# commission = 0
# returns_from_strat = []
# col = []
#
# current_position = 0
# buy_signal = 0
# sell_signal = 0
# hold_time = 0
# for t, ret, pred in zip(test_df_2.index, test_df_2["target_pct"], test_df_2["pred"]):
#     if hold_time > 0:
#         returns_from_strat.append(ret)
#         hold_time = hold_time - 1
#     else:
#         returns_from_strat.append(0)
#
#     if pred:
#         hold_time = 3

#    if buy_signal:
#        returns_from_strat.append(ret - commission)
#        current_position = 1
#        hold_time = 3
#        buy_signal = 0
#    elif sell_signal:
#        returns_from_strat.append(-1*ret - commission)
#        current_position = -1
#        hold_time = 3
#        sell_signal = 0
#    else:
#        returns_from_strat.append(0)
#
#    if current_position == 0 and pred:
#        buy_signal = 1
#        hold_time = 3
#
#    if current_position != 0:
#        hold_time = hold_time - 1
#
#    if hold_time == 1:
#        returns_from_strat.append()



# test_df_2["ret_strat"] = (np.array(returns_from_strat) + 1).cumprod()

#(test_df_2["ret_strat"]*1).plot()
#(test_df_2["pred"]*1).plot()


#test_df_3 = test_df_2.iloc[0:200,:]
# import matplotlib
# plt.pyplot.scatter(x = test_df_2.index, y= test_df_2["ret_strat"], c = test_df_2["pred"])


#test_x.shape

#len(test_main_df)
#len(test_x)
#train_y = np.array(train_y).reshape(-1, 1)
#val_y = np.array(val_y).reshape(-1, 1)
#
#shape_x = np.array(train_x).shape
#shape_y = np.array(val_y).shape
#shape_y = np.array(train_y).shape
#
#
#train_x.shape
#np.array(y_batches).shape
