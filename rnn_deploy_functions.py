import pandas as pd
import numpy as np
import datetime
from rnn_functions import *
from tensorflow.keras.models import load_model
from sklearn.externals import joblib

def query_x(df, datetime_now, x_columns, SEQ_LEN, scaler):
    sub_df = df[df.index <= datetime_now][x_columns]
    not_in_sync = np.abs((sub_df.index[-1] - datetime_now).days) > 1

    if len(sub_df) < SEQ_LEN:
        return None
    elif not_in_sync:
        print("Last date not same:", sub_df.index[-1], "// Current:", datetime_now)
        return None
    else:
        x_df = sub_df[-SEQ_LEN:]
        x = scaler.transform(x_df.values)
        model_input_x = x.reshape((1,x.shape[0],x.shape[1]))
        return model_input_x

def predict_from_model(model_input_x, rnn_model):
    if model_input_x is None:
        print("model_input_x is None! Returning 0!")
        return -1
    else:
        try:
            y = rnn_model.predict(model_input_x)
            return y[0][1]
            # return np.argmax(y)
        except Exception as err:
            print(err)
            return -1
