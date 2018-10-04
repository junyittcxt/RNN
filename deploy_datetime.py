import flask
from flask import request
import pandas as pd
import numpy as np
from functools import reduce
import time
from datetime import datetime
from dateutil import relativedelta
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.externals import joblib
import datetime
from rnn_deploy_functions import *

app = flask.Flask(__name__)

@app.route("/predict_using_datetime", methods=["GET","POST"])
def predict():
    try:
        params = request.args
        yy = int(params.get("year"))
        mm = int(params.get("month"))
        dd = int(params.get("day"))
        hh = int(params.get("hour"))
        mn = int(params.get("minute"))
        ss = int(params.get("second"))

        datetime_now = datetime.datetime(yy,mm,dd,hh,mn,ss)
        model_input_x = query_x(df, datetime_now, x_columns, SEQ_LEN, scaler)
        signal = predict_from_model(model_input_x, rnn_model)
        print(signal)
    except Exception as err:
        print(err)
        signal = -1

    return flask.jsonify(signal_raw = str(signal))


if __name__ == "__main__":
    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


    #Load Model
    # NAME = "SPY-60-5-5-5-0.4-0.0002-256-100-1538115845"
    NAME = "SPY-60-5-5-5-0.4-0.0002-512-100-1538125975"
    folder_dir = f"./RUNS/{NAME}"
    model_input = f"./models/SPY/{NAME}/RNN_Final-053-0.857.model"

    scaler = joblib.load(f"{folder_dir}/rnn_scaler.pkl")
    x_columns = joblib.load(f"{folder_dir}/x_columns.pkl")
    PARAMS_INFO = joblib.load(f"{folder_dir}/PARAMS_INFO.pkl")
    #Load scaler and columns sequence
    SEQ_LEN = int(PARAMS_INFO["SEQ_LEN"])

    rnn_model = load_model(filepath = model_input, custom_objects=None, compile=False)

    df = pd.read_csv("./DATA/x_82_ETF_FOREX_5MIN_RETONLY.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    datetime_now = datetime.datetime(2012,6,20,10,20,20)
    model_input_x = query_x(df, datetime_now, x_columns, SEQ_LEN, scaler)
    predicted_signal = predict_from_model(model_input_x, rnn_model)
    print(datetime_now, predicted_signal)

    datetime_now = datetime.datetime(2013,6,20,10,20,20)
    model_input_x = query_x(df, datetime_now, x_columns, SEQ_LEN, scaler)
    predicted_signal = predict_from_model(model_input_x, rnn_model)
    print(datetime_now, predicted_signal)

    app.run(host='192.168.1.130', port = 3030, debug=True)
