import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base
import math
import numpy as np
import pandas as pd
import functools as ft
import csv
import os
import sklearn.preprocessing as prep
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import copy
from keras import backend as k
from keras.models import load_model
from sklearn import preprocessing

np.set_printoptions(threshold=np.nan)


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 seed=None):

        self.check_data(images, labels)
        seed1, seed2 = random_seed.get_seed(seed)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._total_batches = images.shape[0]

    def check_data(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def total_batches(self):
        return self._total_batches

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._total_batches)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]


        if start + batch_size <= self._total_batches:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


        else:
            self._epochs_completed += 1


            batches_left = self._total_batches - start
            images_left = self._images[start:self._total_batches]
            labels_left = self._labels[start:self._total_batches]


            if shuffle:
                perm = np.arange(self._total_batches)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]


            start = 0
            self._index_in_epoch = batch_size - batches_left
            end = self._index_in_epoch
            images_new = self._images[start:end]
            labels_new = self._labels[start:end]
            return np.concatenate((images_left, images_new), axis=0), np.concatenate((labels_left, labels_new), axis=0)


def load_csv(fname, col_start=2, row_start=0, delimiter=",", dtype=dtypes.float32):
    data = np.genfromtxt(fname, delimiter=delimiter)
    for _ in range(col_start):
        data = np.delete(data, (0), axis=1)
    for _ in range(row_start):
        data = np.delete(data, (0), axis=0)

    return data
def standard_scaler(X_train, X_test):
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

def preprocess_data(stock, seq_len, predict_length=1, split_percent=0.9, problem_class='classification'):
    amount_of_features = len(stock.columns) - 1
    factor_data = stock.as_matrix()
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
        train, result = standard_scaler(train, result)

        X_train = train[:, :-predict_length, :-1]
        y_train = train[:, -1][:, -1]
        X_test = result[int(row):, :-predict_length, :-1]
        y_test = result[int(row):, -1][:, -1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))
        y_train = y_train.reshape(len(y_train),1)
        y_test = y_test.reshape(len(y_test), 1)
        return [X_train, y_train, X_test, y_test]
    else:
        y_train = train[:, -1][:, -1].T

        y_test = result[int(row):, -1][:, -1].T

        train, result = standard_scaler(train, result)

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


def load_data1(path, back_window=128, split_percent=0.9, predict_length=10,problem_class = 'classification'):
    df = pd.read_csv(path,index_col=0)
    df.index = pd.to_datetime(df.index)
    df.dropna(how='any', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    X_train, y_train, X_test, y_test = preprocess_data(df, back_window, predict_length=predict_length, split_percent=split_percent, problem_class=problem_class)
    train = DataSet(X_train, y_train)
    test = DataSet(X_test, y_test)
    if 'classification' in problem_class:
        x_dim = X_train.shape[2]
        y_dim = y_train.shape[1]
        return base.Datasets(train=train, validation=None, test=test),x_dim,y_dim
    elif problem_class == 'regression':
        x_dim = X_train.shape[2]
        y_dim = 1
        return base.Datasets(train=train, validation=None, test=test), x_dim, y_dim
def load_data(path, back_window=128, dim=5, split_percent=0.9, predict_length=10,problem_class = 'classification'):

    def process_data(data):
        stock_set = np.zeros([0, back_window, dim])
        label_set = np.zeros([0, 2])
        for idx in range(data.shape[0] - (back_window + predict_length)):

            stock_set = np.concatenate((stock_set, np.expand_dims(data[range(idx, idx + back_window), :], axis=0)), axis=0)
            if data[idx + (back_window + predict_length), 3] > data[idx + (back_window), 3]:
                lbl = [[1.0, 0.0]]
            else:
                lbl = [[0.0, 1.0]]
            label_set = np.concatenate((label_set, lbl), axis=0)

        return stock_set, label_set
    stocks_set, labels_set = process_data(load_csv(path))



    stocks_set_ = np.zeros(stocks_set.shape)
    for i in range(len(stocks_set)):
        min = stocks_set[i].min(axis=0)
        max = stocks_set[i].max(axis=0)
        stocks_set_[i] = (stocks_set[i] - min) / (max - min)
    stocks_set = stocks_set_
    train_test_idx = int(split_percent * labels_set.shape[0])
    train_stocks = stocks_set[:train_test_idx, :, :]
    train_labels = labels_set[:train_test_idx]
    test_stocks = stocks_set[train_test_idx:, :, :]
    test_labels = labels_set[train_test_idx:]
    train = DataSet(train_stocks, train_labels)
    test = DataSet(test_stocks, test_labels)
    return base.Datasets(train=train, validation=None, test=test)

