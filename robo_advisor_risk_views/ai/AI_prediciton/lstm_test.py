import pandas as pd
from AI_Prediction.Lstm.LSTM import Lstm
from AI_Prediction.Lstm.LSTM import lstm_predict
from Toolbox.wx_talib import *
from Risk_Control.risk_labling import ylzc
from keras.models import load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import math
import os
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from test_plt import test_plt





pred = lstm_predict(data_path='801811.SI.csv',back_window=20,predict_length=1,dropout=0.5,epoch=10,unit=[100],batch_size=200, train=False,problem_class='multi_classification')
print('the pred is',pred)
