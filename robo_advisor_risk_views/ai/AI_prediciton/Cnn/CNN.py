import argparse
import sys
import tensorflow as tf
import functools
from AI_Prediction.Cnn.ops import *
from AI_Prediction.Cnn.loader import *
import pandas as pd
from tensorflow.python.framework import dtypes
import pandas as pd
import os
def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class CNN:
    def __init__(self,
                 image,
                 label,
                 dropout=0.5,
                 conv_size=9,
                 conv_stride=1,
                 ksize=2,
                 pool_stride=2,
                 filter_num=128,
                 padding="SAME",
                 problem_class='classification',
                 learning_rate = 0.001):
        self.image = image
        self.label = label
        self.dropout = dropout
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.ksize = ksize
        self.pool_stride = pool_stride
        self.padding = padding
        self.filter_num = filter_num
        self.problem_class = problem_class
        self.learning_rate = learning_rate
        self.prediction
        self.optimize
        self.accuracy

    @define_scope

    def prediction(self):

        with tf.variable_scope("model") as scope:

            input_image = self.image
            layers = []

            with tf.variable_scope("conv_1"):
                output = relu(conv1d(input_image, self.filter_num, name='conv_1'))
                layers.append(output)

            layer_specs = [
                (self.filter_num * 2, 0.5),
                (self.filter_num * 4, 0.5),
                (self.filter_num * 8, 0.5),
                (self.filter_num * 8, 0.5),
                (self.filter_num * 8, 0.5)
            ]

            for _, (out_channels, dropout) in enumerate(layer_specs):
                with tf.variable_scope("conv_%d" % (len(layers) + 1)):
                    rectified = lrelu(layers[-1], 0.2)

                    convolved = conv1d(rectified, out_channels)

                    output = batchnorm(convolved, is_2d=False)

                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)
                    layers.append(output)
            h_fc1 = relu(fully_connected(layers[-1], 256, name='fc1'))
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)
            out_dim = self.label.shape.as_list()[-1]
            if 'classification' in self.problem_class:
                return tf.sigmoid(fully_connected(h_fc1_drop, output_dim=out_dim, name='fc2'))
            elif self.problem_class == 'regression':
                return tf.sigmoid(fully_connected(h_fc1_drop, output_dim=out_dim, name='fc2'))

    @define_scope
    def optimize(self):
        if 'classification' in self.problem_class:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.prediction))
            return tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)# 最小化用minimize来定义
        elif self.problem_class == 'regression':
            mse = tf.reduce_mean(tf.square(self.label - self.prediction))# 若是回归问题则损失函数应该定义为均方误差最小
            return tf.train.AdamOptimizer(self.learning_rate).minimize(mse)
    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        a = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return a
    @define_scope
    def classification_result(self):
        return tf.argmax(self.prediction, 1)[0]

    @define_scope
    def regression_result(self):
        return self.prediction


def cnn_predict(data_path='',back_window=20, predict_length=10, dropout1=0.3, epoch=1000, train = True, learning_rate=0.001, split_percent=0.85, problem_class = 'classification'):
    db,x_dim,y_dim = load_data1(data_path, back_window=back_window, split_percent=split_percent, predict_length=predict_length,problem_class=problem_class)
    image = tf.placeholder(tf.float32, [None, back_window, x_dim])
    label = tf.placeholder(tf.float32, [None, y_dim])
    dropout = tf.placeholder(tf.float32)
    model = CNN(image, label, dropout=dropout,problem_class = problem_class,learning_rate=learning_rate)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if train==True:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):# 开始进行迭代计算
                images, labels = db.train.next_batch(10)
                if i % 1 == 0:
                    images_eval, labels_eval = db.test.next_batch(10)
                    accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: dropout1})
                    print('step %d, accuracy %g' % (i, accuracy))
                sess.run(model.optimize, {image: images, label: labels, dropout: dropout1})
                if i % epoch == 0:
                    save_path = 'checkpoints/'
                    model_name = 'my_model'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_path_full = os.path.join(save_path, model_name)
                    saver.save(sess, save_path_full)
        else:
            new_saver = tf.train.import_meta_graph('checkpoints/my_model.meta')# 先恢复图结构
            new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))# 再恢复数据

        images_eval, labels_eval = db.test.next_batch(10)

        accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})


        data = pd.read_csv(data_path, index_col=0)
        data.dropna(axis=0, how='any', inplace=True)
        if 'classification' in problem_class:
            latest_pic_data = data.values[-back_window:, :-1]

            scaler = preprocessing.StandardScaler()
            latest_pic_data = scaler.fit_transform(latest_pic_data)

            prediction_result = sess.run(model.classification_result, feed_dict={image: [latest_pic_data], dropout: 1.0})
            return prediction_result
        elif problem_class == 'regression':
            new_batch = data.values[-back_window:, :]
            scaler = preprocessing.StandardScaler()
            batch_scale = scaler.fit(new_batch)
            scaled_data = scaler.transform(new_batch)
            x_new_batch = scaled_data[-back_window:, :-1]
            x_new_batch = x_new_batch.reshape(1, x_new_batch.shape[0], x_new_batch.shape[1])
            pred = sess.run(model.regression_result, feed_dict={image: x_new_batch, dropout: 1.0})
            temp = [0] * new_batch.shape[1]
            temp[-1] = float(pred)
            real_value_pred = batch_scale.inverse_transform([temp])[0][-1]
            return real_value_pred

if __name__ == '__main__':
    pred = cnn_predict(back_window=128, predict_length=10, dropout1=0.3, split_percent=0.9, epoch=10, train = True,learning_rate = 0.001,data_path='F:/Backtest_2018/data/index_data/000001.csv',problem_class = 'regression')
    print(pred)
