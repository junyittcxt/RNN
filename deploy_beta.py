import flask
from flask import request
import pandas as pd
import numpy as np

from functools import reduce
#import custom_functions as cm
import time
from datetime import datetime
from dateutil import relativedelta

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.externals import joblib

def parse_str_data(str_data):
    data_dict = dict()

    for feature in str_data.split(";"):
        feature_data = feature.split(",")
        feature_values = []
        for i, j in enumerate(feature_data):
            if feature_data[0] != "":
                if i == 0:
                    feature_key = j
                else:
                    feature_values.append(float(j))
                data_dict[feature_key] = feature_values
    return data_dict

def loadmodel():
    # import tensorflow as tf
    # filepath = "C:/Users/Workstation/Desktop/rnn_crypto/models/RNN_Final-26-0.571.model"
    # m2 = tf.keras.models.load_model(
    #     filepath = filepath,
    #     custom_objects=None,
    #     compile=False
    # )
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
    return ""

#import MLData_Class as MLData
#import ml_functions as ml
app = flask.Flask(__name__)

@app.route("/predict", methods=["GET","POST"])
def predict():
    try:
        params = request.args
        str_data = str(params.get("data"))
        data_dict = parse_str_data(str_data)

        df_input = pd.DataFrame(data_dict)
        df = df_input[x_columns]
#        temp = range(150)
#        df = pd.DataFrame(dict(btc = temp, vol = temp, a = temp, b = temp, c = temp, d = temp, e = temp, f = temp))
        x_input = df.values
        x_transformed = scaler.transform(x_input)
        x_transformed2 = x_transformed.reshape(1, 150, 8)
        x_transformed2.shape
        predicted_y = rnn_model.predict(x_transformed2)
        signal = np.argmax(np.array(predicted_y))

    except KeyError as err:
        print(err)
        signal = 0
    except ValueError as err:
        print(err)
        signal = 0
    except Exception as err:
        print(err)
        signal = 0

    return flask.jsonify(data)



if __name__ == "__main__":
    #Load Model
    filepath = r'C:/Users/Workstation/Desktop/RNN/models/RNN_Final-23-0.546.h5py'
    rnn_model = load_model(filepath = filepath, custom_objects=None, compile=False)

    #Load scaler and columns sequence
    scaler = joblib.load("C:/Users/Workstation/Desktop/RNN/rnn_scaler.pkl")
    x_columns = joblib.load("C:/Users/Workstation/Desktop/RNN/x_columns.pkl")
    app.run(host='192.168.1.152', port = 3030, debug=True)
