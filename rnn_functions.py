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

import keras.backend as K

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


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

def filter_off_trading_day(df, target, threshold = 0.1):
    df["hh"] = df.index.hour
    df["mm"] = df.index.minute
    df["ss"] = df.index.second
    df["wkday"] = df.index.weekday
    df = df.groupby(["hh", "mm", "ss", "wkday"]).filter(lambda x: np.mean(x[target]!=0) > threshold)
    return df

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

def preprocess_returns_df(df, target_col, SEQ_LEN, scaler = None, fit = True, same_prop = True, shuffle = False):
    x_columns = [c for c in df.columns if c != target_col]
    if scaler is not None:
        if fit:
            scaler.fit(df[x_columns].values)
        df[x_columns] = scaler.transform(df[x_columns].values)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)
    # prev_days = []

    timestamp = []
    for i, j in zip(df.values, df.index):
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            timestamp.append(j)
            # prev_days = []

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

    if shuffle:
        random.shuffle(sequential_data)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    num_example = len(X)
    num_features = len(x_columns)
    return np.array(X), np.array(y), scaler, x_columns, num_example, num_features

def reshape2(x):
    s = [j for j in x.shape]
    x2 = x.reshape(s[0],s[1],s[2],1)

    return x2, s

def preprocess_returns_df_cnn(df, target_col, SEQ_LEN, scaler = None, fit = True, same_prop = True, shuffle = False):
    x_columns = [c for c in df.columns if c != target_col]
    if scaler is not None:
        if fit:
            scaler.fit(df[x_columns].values)
        df.loc[:,x_columns] = scaler.transform(df[x_columns].values)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)
    # prev_days = []

    timestamp = []
    for i, j in zip(df.values, df.index):
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            timestamp.append(j)
            # prev_days = []

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

    if shuffle:
        random.shuffle(sequential_data)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    x_cnn, x_shape = reshape2(np.array(X))

    return x_cnn, np.array(y), timestamp, scaler, x_columns, x_shape

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

def cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME, PATIENCE = None, logs_folder = None, models_folder=None, device_name = "/gpu:0"):
    if logs_folder is None:
        # logs_folder = NAME
        raise Exception("No Logs Folder")
    if models_folder is None:
        raise Exception("No Models Folder")
    if PATIENCE is None:
        PATIENCE = np.round(EPOCHS/3)

    train_x, train_y, val_x, val_y = data
    tf.reset_default_graph()

    with tf.device(device_name):
        model = Sequential()

        # model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences = True))
        # model.add(Dropout(DROPOUT))
        # model.add(BatchNormalization())

        model.add(CuDNNLSTM(NEURONS, input_shape=(train_x.shape[1:])))
        model.add(Dropout(DROPOUT))
        # model.add(BatchNormalization())

        # model.add(Dense(np.round(NEURONS/2,0), activation = "relu"))
        model.add(Dense(1, activation = "sigmoid"))

        opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = 1e-6)
        model.compile(
                      # loss='sparse_categorical_crossentropy',
                      loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy', precision]
                      )

        # logs_dir = f'logs/{logs_folder}'
        # init_dir(logs_dir)

        tensorboard = TensorBoard(log_dir=logs_folder)

        filepath = "RNN_Final-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"
        models_dir = models_folder
        # init_dir(models_dir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint("{}/{}.model".format(models_dir, filepath),
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='min')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=0, mode='min')
        history = model.fit(train_x, train_y,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(val_x, val_y),
                            callbacks=[tensorboard, checkpoint, earlystopping])

    return model, history

def cudnn_lstm_tf_data(dataset_train, dataset_val, DATA_PARAMS, MODEL_PARAMS, logs_folder = None, models_folder=None):
    NUMLAYER = MODEL_PARAMS["NUMLAYER"]
    NUMUNITS = MODEL_PARAMS["NUMUNITS"]
    DROPOUT = MODEL_PARAMS["DROPOUT"]
    LEARNING_RATE = MODEL_PARAMS["LEARNING_RATE"]
    BATCH_SIZE = MODEL_PARAMS["BATCH_SIZE"]
    EPOCHS = MODEL_PARAMS["EPOCHS"]
    NAME = MODEL_PARAMS["NAME"]
    PATIENCE = MODEL_PARAMS["PATIENCE"]
    device_name = "/gpu:0" #+ MODEL_PARAMS["GPU"]

    NUM_FEATURES = DATA_PARAMS["num_features_train"]
    SEQ_LEN = DATA_PARAMS["SEQ_LEN"]
    training_steps = int(np.ceil(DATA_PARAMS["num_example_train"]/BATCH_SIZE))
    val_steps = int(np.ceil(DATA_PARAMS["num_example_val"]/BATCH_SIZE))

    if logs_folder is None:
        # logs_folder = NAME
        raise Exception("No Logs Folder")
    if models_folder is None:
        raise Exception("No Models Folder")
    if PATIENCE is None:
        PATIENCE = np.round(EPOCHS/3)

    tf.reset_default_graph()

    with tf.device(device_name):
        model = Sequential()
        # model.add(CuDNNLSTM(NUMUNITS, input_shape=(train_x.shape[1:])))
        model.add(CuDNNLSTM(NUMUNITS, input_shape=(NUM_FEATURES, SEQ_LEN)))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1, activation = "sigmoid"))
        opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = 1e-6)
        model.compile(
                      loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy', precision]
                      )
        tensorboard = TensorBoard(log_dir=logs_folder)
        filepath = "RNN_Final-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"
        models_dir = models_folder
        checkpoint = tf.keras.callbacks.ModelCheckpoint("{}/{}.model".format(models_dir, filepath),
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='min')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=0, mode='min')
        # history = model.fit(dataset_train.make_one_shot_iterator(),
        history = model.fit(dataset_train.make_one_shot_iterator(),
                            steps_per_epoch=training_steps,
                            # batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=dataset_val.make_one_shot_iterator(),
                            validation_steps=val_steps,
                            callbacks=[tensorboard, checkpoint, earlystopping],
                            verbose=1)

    return model, history
