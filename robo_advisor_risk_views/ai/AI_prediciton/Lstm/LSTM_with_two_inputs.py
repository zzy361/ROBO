import sys
sys.path.append('..')

from AI_toolbox.nlp_data_process.nlp_data_process import *

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute
from keras.layers import Merge, Input, concatenate, average, add
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector, AveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import regularizers
from keras import losses

import matplotlib.pyplot as plt
import seaborn as sns

class Lstm:
    def __init__(self):
        pass

    def build_model(self,factor_dim=5,word2vec_dim=50,layer_neural_num=30,dropout_para=0.3):
        main_input = Input(shape=(30, factor_dim), name='ts_input')
        text_input = Input(shape=(30, word2vec_dim), name='text_input')

        lstm1 = LSTM(layer_neural_num, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(main_input)
        lstm1 = LSTM(layer_neural_num, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(lstm1)
        lstm1 = Flatten()(lstm1)

        lstm2 = LSTM(layer_neural_num, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(text_input)
        lstm2 = LSTM(layer_neural_num, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(lstm2)
        lstm2 = Flatten()(lstm2)

        lstms = concatenate([lstm1, lstm2])

        x1 = Dense(64)(lstms)
        x1 = LeakyReLU()(x1)
        x1 = Dense(1, activation='linear', name='regression')(x1)

        x2 = Dense(64)(lstms)
        x2 = LeakyReLU()(x2)
        x2 = Dropout(dropout_para)(x2)
        x2 = Dense(1, activation='sigmoid', name='class')(x2)

        final_model = Model(inputs=[main_input, text_input], outputs=[x1, x2])
        return final_model

    def data_process(self,factor_data_path,news_data_path,word2vec_dim):
        text_train, text_test = load_text_data(data_path = news_data_path)
        data_chng_train, data_chng_test = load_factor_data(data_path = factor_data_path)

        train_text, test_text = transform_text2sentences(text_train, text_test)

        train_text_vectors, test_text_vectors, model = transform_text_into_vectors(train_text, test_text, word2vec_dim)

        X_train, X_train_text, Y_train, Y_train2 = split_into_XY(data_chng_train, train_text_vectors, 1, 30, 1)
        X_test, X_test_text, Y_test, Y_test2 = split_into_XY(data_chng_test, test_text_vectors, 1, 30, 1)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
        self.data={'X_train':X_train,'Y_train':Y_train,'X_test':X_test,'Y_train2':Y_train2,'Y_test2':Y_test2,'X_train_text':X_train_text,
                   'X_test_text':X_test_text}
        return X_train, X_train_text, X_test, X_test_text,Y_train,Y_test,Y_train2,Y_test2,X_train_text,X_test_text

    def lstm_predict(self):
        model = self.build_model()
        X_train, X_train_text, X_test, X_test_text, Y_train, Y_test, Y_train2, Y_test2, X_train_text, X_test_text = self.data_process()
        opt = Nadam(lr=0.002, clipnorm=0.5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.000001, verbose=1)
        checkpointer = ModelCheckpoint(monitor='val_loss', filepath="model.hdf5", verbose=1, save_best_only=True)
        model.compile(optimizer=opt, loss={'regression': 'mse', 'class': 'binary_crossentropy'}, loss_weights=[1., 0.2])
        try:
            history = model.fit([X_train, X_train_text], [Y_train, Y_train2],
                                      nb_epoch=100,
                                      batch_size=256,
                                      verbose=1,
                                      validation_data=([X_test, X_test_text], [Y_test, Y_test2]),
                                      callbacks=[reduce_lr, checkpointer], shuffle=True)

        except Exception as e:
            print(e)

        finally:
            model.load_weights("model.hdf5")
            pred = model.predict([X_test, X_test_text])[0]

            predicted = pred
            original = Y_test

            plt.title('Actual and predicted')
            plt.legend(loc='best')
            plt.plot(original, color='black', label='Original data')
            plt.plot(pred, color='blue', label='Predicted data')
            plt.show()

            print(np.mean(np.square(predicted - original)))
            print(np.mean(np.abs(predicted - original)))
            print(np.mean(np.abs((original - predicted) / original)))
