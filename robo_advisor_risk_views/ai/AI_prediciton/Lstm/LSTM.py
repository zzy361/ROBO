
import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras import losses
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import copy
from keras import backend as k
from keras.models import load_model
from sklearn import preprocessing


class Lstm():
    def __init__(self):
        pass

    def standard_scaler(self, X_train, X_test):
        train_samples, train_nx, train_ny = X_train.shape
        test_samples, test_nx, test_ny = X_test.shape

        X_train = X_train.reshape((train_samples, train_nx * train_ny))
        X_test = X_test.reshape((test_samples, test_nx * test_ny))
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        X_train = X_train.reshape((train_samples, train_nx, train_ny))
        X_test = X_test.reshape((test_samples, test_nx, test_ny))

        return X_train, X_test

    def accuracy_rate(self, real_seq, prediction_seq):
        num = len(real_seq)
        prediction_seq = prediction_seq.reshape(1, len(prediction_seq))
        real_seq = real_seq.reshape(1, len(real_seq))
        combine = np.concatenate((real_seq, prediction_seq), axis=0).T
        df = pd.DataFrame(combine, columns=['real_seq', 'prediction_seq'])

        df_temp = (df.shift(1) - df)

        accurate_num = 0

        accurate_num += (df_temp[(df_temp['real_seq'] > 0) & (df_temp['prediction_seq'] > 0)]).shape[0]
        accurate_num += (df_temp[(df_temp['real_seq'] <= 0) & (df_temp['prediction_seq'] <= 0)]).shape[0]
        index_list = list(df_temp[(df_temp['real_seq'] > 0) & (df_temp['prediction_seq'] > 0)].index)
        a1 = list(df_temp[(df_temp['real_seq'] < 0) & (df_temp['prediction_seq'] < 0)].index)
        index_list.extend(a1)

        return accurate_num / (num - 1)

    def conjuction_preprocess_data(self, stock, seq_len, predict_length=1, split_percent=0.9,
                                   problem_class='multi_classification'):
        amount_of_features = len(stock.columns) - 1
        factor_data = stock.as_matrix()
        scaler_fitter = prep.StandardScaler().fit(factor_data)
        factor_data = scaler_fitter.transform(factor_data)
        sequence_length = seq_len + predict_length
        result = []
        for index in range(len(factor_data) - sequence_length):
            result.append(factor_data[index: index + sequence_length])

        result = np.array(result)
        index = list(range(len(result)))
        np.random.shuffle(index)
        result = result[index]

        row = round(split_percent * result.shape[0])
        train = result[: int(row), :]

        index = list(range(len(train)))
        np.random.shuffle(index)
        train = train[index]
        if problem_class == 'regression':

            X_train = train[:, :-predict_length, :-1]
            y_train = train[:, -1][:, -1]
            X_test = result[int(row):, :-predict_length, :-1]
            y_test = result[int(row):, -1][:, -1]

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))
            return [X_train, y_train, X_test, y_test, scaler_fitter]
        else:
            y_train = train[:, -1][:, -1].T

            self.weights = np.array([0.4, 0.4])
            y_test = result[int(row):, -1][:, -1].T

            X_train = train[:, :-predict_length, :-1]

            X_test = result[int(row):, :-predict_length, : -1]

            encoder = LabelEncoder()
            encoded_y_train = encoder.fit_transform(y_train)

            y_train = np_utils.to_categorical(encoded_y_train)
            encoded_y_test = encoder.fit_transform(y_test)

            y_test = np_utils.to_categorical(encoded_y_test)


            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

            return [X_train, y_train, X_test, y_test]

    def preprocess_data(self, stock, seq_len, predict_length=1, split_percent=0.9, problem_class='multi_classification',
                        class_num=2, future_process=False):
        amount_of_features = len(stock.columns) - 1
        factor_data = stock.as_matrix()


        if future_process:
            sequence_length = seq_len + predict_length
        else:
            sequence_length = seq_len
        result = []
        for index in range(len(factor_data) - sequence_length):
            result.append(factor_data[index: index + sequence_length])

        result = np.array(result)


        row = round(split_percent * result.shape[0])
        train = result[: int(row), :]

        index = list(range(len(train)))
        np.random.shuffle(index)
        train = train[index]
        if problem_class == 'regression':
            train, result = self.standard_scaler(train, result)

            X_train = train[:, :-predict_length, :-1]
            y_train = train[:, -1][:, -1]
            X_test = result[int(row):, :-predict_length, :-1]
            y_test = result[int(row):, -1][:, -1]

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))
            return [X_train, y_train, X_test, y_test]
        else:
            y_train = train[:, -1][:, -1].T

            self.weights = np.array([0.4, 0.4])
            y_test = result[int(row):, -1][:, -1].T

            train, result = self.standard_scaler(train, result)

            if future_process:
                X_train = train[:, :-predict_length, :-1]

                X_test = result[int(row):, :-predict_length, : -1]
            else:
                X_train = train[:, :, :-1]

                X_test = result[int(row):, :, : -1]

            encoder = LabelEncoder()
            encoded_y_train = encoder.fit_transform(y_train)

            y_train = np_utils.to_categorical(encoded_y_train, num_classes=class_num)
            encoded_y_test = encoder.fit_transform(y_test)
            y_test = np_utils.to_categorical(encoded_y_test, num_classes=class_num)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

            return [X_train, y_train, X_test, y_test]

    def weighted_loss(self, y_true, y_pred):
        result = -k.sum(y_true * k.log(y_pred) * self.weights, axis=1)
        return result

    def build_model(self, layers, units, dropout=0.3, problem_class='regression', class_num=2):
        model = Sequential()

        model.add(LSTM(units=units[0], input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense(units=40))
        model.add(Dropout(dropout))
        model.add(Dense(units=20))
        model.add(Dropout(dropout))


        start = time.time()
        if problem_class == 'regression':
            model.add(Dense(units=layers[-1]))
            model.add(Activation("linear"))
            model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
        elif problem_class == 'binary_classification':
            model.add(Dense(units=class_num))
            model.add(Activation("sigmoid"))
            model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        elif problem_class == 'multi_classification':
            model.add(Dense(units=class_num))
            model.add(Activation("softmax"))
            model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
        print("Compilation Time : ", time.time() - start)
        return model


def lstm_predict(data_path, back_window, predict_length, dropout, epoch, batch_size=100, unit=[100], train=True,
                 split_percent=0.85, problem_class='classification'):
    lstm = Lstm()
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.dropna(how='any', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    X_train, y_train, X_test, y_test = lstm.preprocess_data(df, back_window, predict_length=predict_length,
                                                            split_percent=split_percent, problem_class=problem_class)

    if train == True:
        if 'classification' in problem_class:
            model = lstm.build_model([X_train.shape[2], back_window, len(set(df.iloc[:, -1].values))], units=unit,
                                     dropout=dropout, problem_class=problem_class)

            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                nb_epoch=epoch,
                validation_split=0.1,
                shuffle=True,
                verbose=2)
            pred = model.predict(X_test)
            model.save('prediction.h5')
            return pred
        elif problem_class == 'regression':
            model = lstm.build_model([X_train.shape[2], back_window, 1], units=unit, dropout=dropout,
                                     problem_class=problem_class)
            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                nb_epoch=epoch,
                validation_split=0.1,
                shuffle=True,
                verbose=1)
            pred = model.predict(X_test)
            model.save('prediction.h5')
            return pred
    else:
        if 'classification' in problem_class:
            model = load_model('prediction.h5')
            last_data = X_test[-1]
            new_batch = df.iloc[-back_window - 1:, :]
            scaler = preprocessing.StandardScaler()
            batch_scale = scaler.fit(new_batch)
            scaled_data = scaler.transform(new_batch)
            x_new_batch = scaled_data[-back_window:, :-1]
            x_new_batch = x_new_batch.reshape(1, x_new_batch.shape[0], x_new_batch.shape[1])
            pred = model.predict(x_new_batch)
            return pred[0]
        elif problem_class == 'regression':
            model = load_model('prediction.h5')

            new_batch = df.iloc[-back_window - 1:, :]
            scaler = preprocessing.StandardScaler()
            batch_scale = scaler.fit(new_batch)
            scaled_data = scaler.transform(new_batch)
            x_new_batch = scaled_data[-back_window:, :-1]
            x_new_batch = x_new_batch.reshape(1, x_new_batch.shape[0], x_new_batch.shape[1])
            pred = model.predict(x_new_batch)
            temp = [0] * new_batch.shape[1]
            temp[-1] = pred
            real_value_pred = batch_scale.inverse_transform([temp])[0][-1]
            return real_value_pred


if __name__ == '__main__':
    lstm = LSTM()
    df = pd.read_csv('000002-from-1995-01-01.csv')
    window = 20
    X_train, y_train, X_test, y_test = lstm.preprocess_data(df[:: -1], window, predict_length=1, split_percent=0.85)

    model = lstm.build_model([X_train.shape[2], window, 100, 1], dropout=0.3, problem_class='classification')
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y_train)

    dummy_y = np_utils.to_categorical(encoded_Y)
    model.fit(
        X_train,
        dummy_y,
        batch_size=768,
        nb_epoch=10,
        validation_split=0.1,
        verbose=1)
    diff = []
    ratio = []
    pred = model.predict(X_test)
    for u in range(len(y_test)):
        pr = pred[u][0]
        ratio.append((y_test[u] / pr) - 1)
        diff.append(abs(y_test[u] - pr))

    import matplotlib.pyplot as plt2

    print(lstm.accuracy_rate(y_test, pred))
    plt2.plot(pred, color='red', label='Prediction')
    plt2.plot(y_test, color='blue', label='Ground Truth')
    plt2.legend(loc='upper left')
    plt2.show()
