import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from collections import deque
import random
import os

from sklearn.metrics import confusion_matrix

def cumulative_returns(returns):
    return np.prod(1+np.array(returns)) - 1

def performance_binary(x, y, model, sample_type = "Sample", threshold = 0.5, silence = False):
    y_pred = model.predict(x)[:,1]
    y_actual = y
    try:
        y_pred_class = np.array(y_pred) >= threshold
        y_test_class = np.array(y_actual) >= threshold
        cm = confusion_matrix(y_test_class, y_pred_class)
        accuracy = (cm[1,1]+cm[0,0])/np.sum(cm)
        precision = cm[1,1]/(cm[1,1]+cm[0,1])
        sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
        class_balance = np.sum(y_actual)/len(y_actual)
        if not silence:
            print('Confusion Matrix')
            print(cm)
            print("Accuracy: " + str(accuracy))
            print("Positive Predictive Value/Precision: " + str(precision))
            print("True Positive Rate/Sensitivity: " + str(sensitivity))
            print("Proportion of Positive: " + str(class_balance))
            print("Excess Positive Predictive Value: " + str(precision-class_balance))

        performance_dict = dict(tn = cm[0,0], fn = cm[1,0],
                                tp = cm[1,1], fp = cm[0,1],
                                accuracy = accuracy,
                                precision = precision,
                                sensitivity = sensitivity,
                                proportion_positive = class_balance,
                                excess_precision = precision-class_balance,
                                sample = sample_type
                                )
    except:
        return dict()

    return performance_dict

def create_target(df, target_col, FUTURE_PERIOD_PREDICT, FUNC = cumulative_returns):
    df['target'] = df[target_col].rolling(window = FUTURE_PERIOD_PREDICT).apply(lambda x: FUNC(x))
    df['target'] = df['target'].shift(-FUTURE_PERIOD_PREDICT)
    df = df.dropna()
    return df

def classify_returns(returns, threshold = 0, flip = False):
    if flip:
        if returns < threshold:
            return 1
        else:
            return 0
    else:
        if returns > threshold:
            return 1
        else:
            return 0


def classify_target(df, target_col = "target", threshold = 0, flip = False):
    df[target_col] = df[target_col].apply(lambda x: classify_returns(x, threshold, flip))
    return df


def sample_cc_main_df(RATIO_TO_PREDICT, FUTURE_PERIOD_PREDICT):
    colnames = ["time", "low", "high", "open", "close", "volume"]
    ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

    main_df = pd.DataFrame()

    for ratio in ratios:
        dataset = f"Sample_Data/{ratio}.csv"
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

    return main_df

def split_df(main_df, proportion = [0.10, 0.25]):
    p1, p2 = proportion
    times = sorted(main_df.index.values)
    last_5pct = times[-int(p1*len(times))]
    last_25pct = times[-int(p2*len(times))]

    training_main_df = main_df[(main_df.index < last_25pct)]
    validation_main_df = main_df[(main_df.index >= last_25pct) & (main_df.index < last_5pct)]
    test_main_df = main_df[(main_df.index >= last_5pct)]
    return training_main_df, validation_main_df, test_main_df

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# def split_df_by_one_timepoint(df, time_point):


def split_df_by_time(df, time_points = []):
    if len(time_points) == 0:
        return [df]
    else:
        # time_points = ["20080505", "20080507", "20080508"]
        t = [range(len(df.index))[np.sum((df.index < timepoint))] for timepoint in time_points]
        df_list = np.split(df, t, axis = 0)
        return df_list

def split_df_by_prop(df, prop = []):
    if len(prop)==0:
        return [df]
    else:
        T = [int(j) for j in ((len(df))*np.array(prop))]
        df_list = np.split(df, T, axis = 0)
        return df_list

def preprocess_returns_df(df, target_col, SEQ_LEN, scaler = None, fit = True, same_prop = True):
    x_columns = [c for c in df.columns if c != target_col]
    if scaler is not None:
        if fit:
            scaler.fit(df[x_columns].values)
        df[x_columns] = scaler.transform(df[x_columns].values)

    sequential_data = []
    # prev_days = deque(maxlen = SEQ_LEN)
    prev_days = []

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            prev_days = []

    random.shuffle(sequential_data)

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

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y), scaler, x_columns

def preprocess_df(df, scaler, SEQ_LEN, fit = True, same_prop = False):
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

    timestamp = []
    for i,j in zip(df.values, df.index):
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            timestamp.append(j)

    random.shuffle(sequential_data)

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

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

#    return np.array(X), np.array(y), df
    return np.array(X), np.array(y), np.array(timestamp), df, scaler, x_columns

def init_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME, logs_folder = None, models_folder=None, device_name = "/gpu:1"):
    if logs_folder is None:
        logs_folder = NAME
    if models_folder is None:
        models_folder = NAME

    train_x, train_y, val_x, val_y = data
    tf.reset_default_graph()

    with tf.device(device_name):
        model = Sequential()

        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())


        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        # model.add(LSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        # model.add(LSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:])))
        # model.add(LSTM(NEURONS, input_shape=(train_x.shape[1:])))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())

        model.add(Dense(np.round(NEURONS/2,0), activation = "relu"))
        model.add(Dense(2, activation = "softmax"))

        opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = 1e-6)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy']
                      )

        logs_dir = f'logs/{logs_folder}'
        init_dir(logs_dir)

        tensorboard = TensorBoard(log_dir=f'{logs_dir}/{NAME}')

        filepath = "RNN_Final-{epoch:03d}-{val_acc:.3f}"
        models_dir = f'models/{models_folder}/{NAME}'
        init_dir(models_dir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint("{}/{}.model".format(models_dir, filepath),
                                                           monitor='val_acc',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='max')

        history = model.fit(train_x, train_y,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(val_x, val_y),
                            callbacks=[tensorboard, checkpoint])

    return model, history
