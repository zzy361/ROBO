import pandas as pd
from AI_Prediction.Lstm.LSTM import Lstm
from Toolbox.wx_talib import *
from Risk_Control.risk_labling import ylzc
from keras.models import load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import math
import os
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping
from test_plt import test_plt
original_data = pd.read_csv('801811.SI.csv', index_col=0)
original_data = MA(original_data, 5)
original_data = MA(original_data, 10)
original_data = MA(original_data, 22)
original_data = MA(original_data, 66)
original_data = MA(original_data, 125)
original_data = MA(original_data, 250)

original_data = MACD(original_data, 3, 8)
original_data = MACD(original_data, 10, 20)
original_data = MACD(original_data, 10, 40)
original_data = MACD(original_data, 30, 60)
original_data = MACD(original_data, 50, 100)
original_data = VORTEX(original_data, 10)
original_data = ULTOSC(original_data)
original_data = TSI(original_data, 15, 5)
original_data = TRIX(original_data, 10)
original_data = STOK(original_data)
original_data = STDDEV(original_data, 10)
original_data = STDDEV(original_data, 20)
original_data = STDDEV(original_data, 40)
original_data = RSI100(original_data, 10)

original_data = MASS(original_data)
original_data = MOM(original_data, 15)
window = 60
raw = pd.read_csv('raw.csv', index_col=0)
original_data['label'] = raw['label']
original_data.iloc[0, -1] = 1
original_data['label'].fillna(0, inplace=True)

original_data.dropna(axis=0, how='any', inplace=True)


original_data.to_csv('p.csv')



lstm = Lstm()
X_train, y_train, X_test, y_test = lstm.preprocess_data(original_data, window, predict_length=0, split_percent=0.76)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
label_dim = len(set(original_data['label'].values))

clf = DecisionTreeClassifier(max_depth=12)

clf.fit(X, y)

pred = model.predict(X_test)
a = pred.argmax(axis=1)
a = a.astype(np.int16)
pp = pd.read_csv('pp.csv', index_col=0)

pp.iloc[-len(a):, -1] = a

b=y_test.argmax(axis=1)
np.savetxt('a.txt', a)
np.savetxt('b.txt', b)
print(pred.argmax(axis=1))

print(y_test.argmax(axis=1))

test_plt(pp,len(a),label_dim-1)
