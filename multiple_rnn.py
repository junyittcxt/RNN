import optparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os, time, sys, sklearn
from sklearn.externals import joblib
from rnn_functions import *
import time

def reshape2d(z):
    z = z.reshape((z.shape[0], -1))
    return z

def get_num_features(x_columns_file):
    x_columns = joblib.load(x_columns_file)
    return len(x_columns)

def tfdata_generator(file_x, file_y, is_training, num_features = 54, SEQ_LEN = 60, batch_size=128):
    def preprocess_fn(features, label):
        # x = tf.reshape(tf.cast(features, tf.float32), (1, num_features, SEQ_LEN))
        x = tf.reshape(tf.cast(features, tf.float32), (num_features, SEQ_LEN))
        y = label
        return x, y

    filenames = [file_x, file_y]
    record_defaults_x = [tf.float32] * num_features*SEQ_LEN
    record_defaults_y = [tf.float32] * 1
    dataset_train_x = tf.contrib.data.CsvDataset(file_x,record_defaults_x)
    dataset_train_y = tf.contrib.data.CsvDataset(file_y,record_defaults_y)
    dataset = tf.data.Dataset.zip((dataset_train_x, dataset_train_y))
    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=4,  # cpu cores
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset

def init_model_dir(DATA_PARAMS, MODEL_PARAMS):
    model_dir = os.path.join("output",DATA_PARAMS["TARGET_TO_PREDICT"],MODEL_PARAMS["NAME"])
    checkpoint_dir = os.path.join(model_dir, 'ModelCheckpoints')
    tensorboard_dir = os.path.join(model_dir, 'TensorBoardLogs')
    init_dir(model_dir)
    init_dir(checkpoint_dir)
    init_dir(tensorboard_dir)

    return model_dir, checkpoint_dir, tensorboard_dir


def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-p", "--preproc", default=0, help="Force Preprocess")

    optparser.add_option("-s", "--seqlen", default=60, help="DATA: SEQ_LEN")
    optparser.add_option("-a", "--asset", default="SPY", help="DATA,MODEL: TARGET_TO_PREDICT")
    optparser.add_option("-r", "--nrows", default=1e5, help="DATA: NROWS")
    optparser.add_option("-f", "--futureperiod", default=3, help="DATA: FUTURE_PERIOD_PREDICT")

    optparser.add_option("-d", "--dropout", default=0, help="MODEL: DROPOUT")
    optparser.add_option("-e", "--epochs", default=0, help="MODEL: EPOCHS")
    optparser.add_option("-b", "--batchsize", default=128, help="MODEL: BATCH_SIZE")
    optparser.add_option("-l", "--learningrate", default=0.00005, help="MODEL: LEARNING_RATE")
    optparser.add_option("-n", "--numunits", default=5, help="MODEL: NUMUNITS")

    optparser.add_option("-g", "--gpu", default="0", help="MODEL: GPU 0 or 1 (2080 or 1050)")
    opts = optparser.parse_args()[0]

    #TENSORFLOW CONFIGURATION
    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
    os.environ["CUDA_VISIBLE_DEVICES"]= opts.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    ##INIT PARAMS
    DATA_PARAMS = dict()
    MODEL_PARAMS = dict()
    DATA_PARAMS = dict(NROWS = int(opts.nrows), TARGET_TO_PREDICT = opts.asset, SEQ_LEN = int(opts.seqlen), FUTURE_PERIOD_PREDICT = int(opts.futureperiod),
    )
    DATA_PARAMS["DATANAME"] = f'{DATA_PARAMS["TARGET_TO_PREDICT"]}-{DATA_PARAMS["SEQ_LEN"]}-{DATA_PARAMS["FUTURE_PERIOD_PREDICT"]}'
    MODEL_PARAMS = dict(NUMLAYER = 1, NUMUNITS = int(opts.numunits), DROPOUT = float(opts.dropout), LEARNING_RATE = float(opts.learningrate),
                        BATCH_SIZE = int(opts.batchsize), EPOCHS = int(opts.epochs), PATIENCE = int(opts.epochs), GPU = opts.gpu
    )
    MODEL_PARAMS["NAME"] = f'{DATA_PARAMS["TARGET_TO_PREDICT"]}-{DATA_PARAMS["SEQ_LEN"]}-{DATA_PARAMS["FUTURE_PERIOD_PREDICT"]}-{MODEL_PARAMS["NUMLAYER"]}-{MODEL_PARAMS["NUMUNITS"]}-{MODEL_PARAMS["DROPOUT"]}-{MODEL_PARAMS["LEARNING_RATE"]}-{MODEL_PARAMS["BATCH_SIZE"]}-{MODEL_PARAMS["EPOCHS"]}-{int(time.time())}'


    ##PREPROCESS
    preprocess_t0 = time.time()
    asset_data_dir = 'output/{tp}/DATA/{dataname}'.format(tp = DATA_PARAMS['TARGET_TO_PREDICT'], dataname = DATA_PARAMS["DATANAME"])
    if int(opts.preproc) == 1:
        print("ENTER PREPROCESS...")
        pre_proc_and_dump_data(DATA_PARAMS)
    else:
        print("WAITING FOR PREPROCESS...")
        num_data_files =  0
        if os.path.isdir(asset_data_dir):
            num_data_files =  len(os.listdir(asset_data_dir))
        while num_data_files < 7:
            if os.path.isdir(asset_data_dir):
                num_data_files =  len(os.listdir(asset_data_dir))
            preprocess_t1 = time.time()
            print("WAITING FOR PREPROCESS...")
            time.sleep(10)
            if preprocess_t1 - preprocess_t0 > 60*30:
                raise Exception("Terminating: Preprocessing too long! >30 Minutes!")

    ##TRAINING
    file_train_x = os.path.join(asset_data_dir, "train_x.csv")
    file_train_y = os.path.join(asset_data_dir, "train_y.csv")
    file_val_x = os.path.join(asset_data_dir, "val_x.csv")
    file_val_y = os.path.join(asset_data_dir, "val_y.csv")

    DATA_PARAMS_file = os.path.join(asset_data_dir, "DATA_PARAMS.pkl")
    DATA_PARAMS = joblib.load(DATA_PARAMS_file)

    dataset_train = tfdata_generator(file_train_x, file_train_y, is_training = True, num_features = int(DATA_PARAMS["num_features_train"]), SEQ_LEN=DATA_PARAMS["SEQ_LEN"], batch_size=MODEL_PARAMS["BATCH_SIZE"])
    dataset_val = tfdata_generator(file_val_x, file_val_y, is_training = False, num_features = int(DATA_PARAMS["num_features_train"]), SEQ_LEN=DATA_PARAMS["SEQ_LEN"], batch_size=MODEL_PARAMS["BATCH_SIZE"])

    model_dir, checkpoint_dir, tensorboard_dir = init_model_dir(DATA_PARAMS, MODEL_PARAMS)
    t0 = time.time()
    model, history = cudnn_lstm_tf_data(dataset_train, dataset_val, DATA_PARAMS, MODEL_PARAMS, logs_folder = tensorboard_dir, models_folder=checkpoint_dir)
    # data = (train_x, train_y, val_x, val_y)
    # model, history = cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME, PATIENCE, logs_folder = tensorboard_dir, models_folder=checkpoint_dir, device_name = "/gpu:0")
    t1 = time.time()
    print("Training: Done! Time Elapsed:", str(t1-t0), "seconds!")

    print(DATA_PARAMS)
    print(MODEL_PARAMS)
    # performance_binary(train_x, train_y, model, sample_type = "train")
    # performance_binary(val_x, val_y, model, sample_type = "val")
    # performance_binary(test_x, test_y, model, sample_type = "test")


def pre_proc_and_dump_data(DATA_PARAMS):
        NROWS = DATA_PARAMS["NROWS"]
        TARGET_TO_PREDICT = DATA_PARAMS["TARGET_TO_PREDICT"]
        FUTURE_PERIOD_PREDICT = DATA_PARAMS["FUTURE_PERIOD_PREDICT"]
        SEQ_LEN = DATA_PARAMS["SEQ_LEN"]

        #Load
        if NROWS <= 0:
            df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv")
        else:
            df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv", nrows = NROWS)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        print("Load: Done!")

        #Pre-process (target)
        df = filter_off_trading_day(df, target = TARGET_TO_PREDICT, threshold = 0.1)
        df = create_target(df, TARGET_TO_PREDICT, FUTURE_PERIOD_PREDICT, cumulative_returns)
        df = classify_target(df, "target", 0.000, False)
        print("Pre-Process: Done!")

        #Split
        prop = [0.6, 0.75, 0.85]
        df_list_2 = split_df_by_prop(df, prop = prop)
        startdates = [j.index[0] for j in df_list_2]
        enddates = [j.index[-1] for j in df_list_2]

        print([j.shape for j in df_list_2])
        print("startdates:", startdates)
        print("enddates:", enddates)
        DATA_PARAMS["startdates"] = startdates
        DATA_PARAMS["enddates"] = enddates
        print("Split: Done!")

        #Scaling
        scaler = sklearn.preprocessing.StandardScaler()
        train_x, train_y, scaler, x_columns, num_example_train, num_features_train = preprocess_returns_df(df=df_list_2[0], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = True, same_prop = True, shuffle = True)
        val_x, val_y, _, _, num_example_val, num_features_val = preprocess_returns_df(df=df_list_2[1], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = True, shuffle = True)
        # test_x, test_y, _, _ = preprocess_returns_df(df=df_list_2[2], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = False, shuffle = False)
        # test2_x, test2_y, _, _ = preprocess_returns_df(df=df_list_2[3], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = False, shuffle = False)
        DATA_PARAMS["num_example_train"] = num_example_train
        DATA_PARAMS["num_features_train"] = num_features_train
        DATA_PARAMS["num_example_val"] = num_example_val
        DATA_PARAMS["num_features_val"] = num_features_val

        print("Scaling: Done!")

        #INIT DIR AND DUMP DATA
        asset_data_dir = 'output/{tp}/DATA/{dataname}'.format(tp = DATA_PARAMS['TARGET_TO_PREDICT'], dataname = DATA_PARAMS["DATANAME"])
        init_dir(asset_data_dir)

        #SAVING
        #save to csv

        # np.save(os.path.join(asset_data_dir, "train_x.dat"), train_x)
        # np.save(os.path.join(asset_data_dir, "train_y.dat"), train_y)
        # np.save(os.path.join(asset_data_dir, "val_x.dat"), val_x)
        # np.save(os.path.join(asset_data_dir, "val_y.dat"), val_y)
        print(val_y)
        np.savetxt(os.path.join(asset_data_dir, "train_x.csv"), reshape2d(train_x), delimiter=",")
        np.savetxt(os.path.join(asset_data_dir, "train_y.csv"), train_y, delimiter=",")
        np.savetxt(os.path.join(asset_data_dir, "val_x.csv"), reshape2d(val_x), delimiter=",")
        np.savetxt(os.path.join(asset_data_dir, "val_y.csv"), val_y, delimiter=",")
        joblib.dump(scaler, os.path.join(asset_data_dir,'rnn_scaler.pkl'))
        joblib.dump(x_columns, os.path.join(asset_data_dir,'x_columns.pkl'))
        joblib.dump(DATA_PARAMS, os.path.join(asset_data_dir,'DATA_PARAMS.pkl'))
        print("Saving: Done!")





if __name__ == "__main__":
    main()
