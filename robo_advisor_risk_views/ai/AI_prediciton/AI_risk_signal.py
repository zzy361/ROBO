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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from test_plt import test_plt,test_plt_gif
original_data = pd.read_csv('801811.SI.csv', index_col=0)
original_data = MA(original_data, 5)

original_data = MA(original_data, 22)
original_data = MA(original_data, 66)
original_data = MA(original_data, 125)
original_data = MA(original_data, 250)
original_data = FORCE(original_data, 10)
original_data = BBANDS(original_data, 10)
original_data = MACD(original_data, 3, 8)
original_data = MACD(original_data, 10, 20)
original_data = MACD(original_data, 10, 40)
original_data = MACD(original_data, 30, 60)

original_data = TSI(original_data, 15, 5)
original_data = TRIX(original_data, 10)

original_data.dropna(axis=0,how='any',inplace=True)
original_data.to_csv('original.csv')

window = 30
raw = pd.read_csv('raw.csv', index_col=0)
original_data['label'] = raw['label']
original_data.iloc[0, -1] = 1
original_data['label'].fillna(0, inplace=True)

original_data.dropna(axis=0, how='any', inplace=True)


original_data.to_csv('p.csv')



lstm = Lstm()
X_train, y_train, X_test, y_test = lstm.preprocess_data(original_data, window, predict_length=50, split_percent=0.76,class_num=3)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
label_dim = len(set(original_data['label'].values))
model = lstm.build_model([X_train.shape[2], window, 20, label_dim], units=[200], dropout=0.3, problem_class='multi_classification')

filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=150)

callbacks_list = [checkpoint]
if os.path.exists('risk_predict.h5'):
    model = load_model('risk_predict.h5')
else:
    history = model.fit(
        X_train,
        y_train,
        batch_size=1000,
        nb_epoch=200,
        validation_split=0.1,
        verbose=2,
        shuffle=True,
        callbacks=callbacks_list)

    model.save('risk_predict.h5')
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])


pred = model.predict(X_test)
a = pred.argmax(axis=1)
a = a.astype(np.int16)
pp = pd.read_csv('pp.csv', index_col=0)


b = y_test.argmax(axis=1)
np.savetxt('a.txt', a)
np.savetxt('b.txt', b)
print(pred.argmax(axis=1))

print(y_test.argmax(axis=1))

test_plt_gif(pp, len(a), label_dim - 1)
