import optparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os, time, sys, sklearn
from sklearn.externals import joblib
from rnn_functions import *
import time

def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-p", "--preproc", default=0, help="Force Preprocess")

    optparser.add_option("-s", "--seqlen", default=60, help="DATA: SEQ_LEN")
    optparser.add_option("-a", "--asset", default="SPY", help="DATA,MODEL: TARGET_TO_PREDICT")
    optparser.add_option("-r", "--nrows", default=1e6, help="DATA: NROWS")
    optparser.add_option("-f", "--futureperiod", default=3, help="DATA: FUTURE_PERIOD_PREDICT")

    optparser.add_option("-d", "--dropout", default=0, help="MODEL: DROPOUT")
    optparser.add_option("-b", "--batchsize", default=256, help="MODEL: BATCH_SIZE")
    optparser.add_option("-l", "--learningrate", default=0.0001, help="MODEL: LEARNING_RATE")
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
                        BATCH_SIZE = int(opts.batchsize), EPOCHS = 200, PATIENCE = 200
    )
    MODEL_PARAMS["NAME"] = f'{DATA_PARAMS["TARGET_TO_PREDICT"]}-{DATA_PARAMS["SEQ_LEN"]}-{DATA_PARAMS["FUTURE_PERIOD_PREDICT"]}-{MODEL_PARAMS["NUMLAYER"]}-{MODEL_PARAMS["NUMUNITS"]}-{MODEL_PARAMS["DROPOUT"]}-{MODEL_PARAMS["LEARNING_RATE"]}-{MODEL_PARAMS["BATCH_SIZE"]}-{MODEL_PARAMS["EPOCHS"]}-{int(time.time())}'


    ##PREPROCESS
    asset_data_dir = 'output/{tp}/DATA/{dataname}'.format(tp = DATA_PARAMS['TARGET_TO_PREDICT'], dataname = DATA_PARAMS["DATANAME"])
    if not os.path.isdir(asset_data_dir) and int(opts.preproc) == 1:
        print("ENTER PREPROCESS...")
        pre_proc_and_dump_data(DATA_PARAMS)
    else:
        num_data_files =  0
        print("WAITING FOR PREPROCESS...")
        while num_data_files != 7:
            time.sleep(10)
            if os.path.isdir(asset_data_dir):
                num_data_files =  len([name for name in os.listdir(asset_data_dir) if os.path.isfile(name)])

    ##TRAINING
    # t0 = time.time()
    # data = (train_x, train_y, val_x, val_y)
    # model, history = cudnn_lstm(data, NUMLAYER, NEURONS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, NAME, PATIENCE, logs_folder = tensorboard_dir, models_folder=checkpoint_dir, device_name = "/gpu:0")
    # t1 = time.time()
    # print("Training: Done! Time Elapsed:", str(t1-t0), "seconds!")

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
        if NROWS is None:
            df = pd.read_csv("./DATA/x_82_ETF_FOREX_MIN_RETONLY.csv")
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
        train_x, train_y, scaler, x_columns = preprocess_returns_df(df=df_list_2[0], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = True, same_prop = True, shuffle = True)
        val_x, val_y, _, _ = preprocess_returns_df(df=df_list_2[1], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = True, shuffle = False)
        # test_x, test_y, _, _ = preprocess_returns_df(df=df_list_2[2], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = False, shuffle = False)
        # test2_x, test2_y, _, _ = preprocess_returns_df(df=df_list_2[3], target_col = "target", scaler = scaler, SEQ_LEN = SEQ_LEN, fit = False, same_prop = False, shuffle = False)
        print("Scaling: Done!")

        #INIT DIR AND DUMP DATA
        asset_data_dir = 'output/{tp}/DATA/{dataname}'.format(tp = DATA_PARAMS['TARGET_TO_PREDICT'], dataname = DATA_PARAMS["DATANAME"])
        init_dir(asset_data_dir)

        #SAVING
        np.save(os.path.join(asset_data_dir, "train_x.npy"), train_x)
        np.save(os.path.join(asset_data_dir, "train_y.npy"), train_y)
        np.save(os.path.join(asset_data_dir, "val_x.npy"), val_x)
        np.save(os.path.join(asset_data_dir, "val_y.npy"), val_y)
        joblib.dump(scaler, os.path.join(asset_data_dir,'rnn_scaler.pkl'))
        joblib.dump(x_columns, os.path.join(asset_data_dir,'x_columns.pkl'))
        joblib.dump(DATA_PARAMS, os.path.join(asset_data_dir,'DATA_PARAMS.pkl'))
        print("Saving: Done!")


def init_dir_dump_scaler_params(PARAMS_INFO, scaler, x_columns):
    run_dir = f'output/{PARAMS_INFO["TARGET_TO_PREDICT"]}/{PARAMS_INFO["NAME"]}'
    checkpoint_dir = f'{run_dir}/ModelCheckpoints'
    tensorboard_dir = f'{run_dir}/TensorBoardLogs'
    init_dir(run_dir)
    init_dir(checkpoint_dir)
    init_dir(tensorboard_dir)

    print("Meta-data Dump: Done!")

if __name__ == "__main__":
    main()
