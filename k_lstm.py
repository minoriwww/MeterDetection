# -*- coding: utf-8 -*-
# run in py3 !!
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1";

import tensorflow as tf

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True
tf.Session(config=config)

import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import time
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras import backend as K
import keras.layers.convolutional as conv
from keras.layers import merge
from keras.wrappers.scikit_learn import KerasRegressor
from keras import utils
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.layers import pooling
from keras.models import Sequential, Model
from keras.regularizers import l1, l2
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution1D, Convolution2D, LSTM
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.callbacks import EarlyStopping
from keras import callbacks
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from keras.models import Model

from keras import initializers, layers
from keras.optimizers import SGD, Adadelta, Adam
from keras.regularizers import l1, l2
from keras import regularizers
import sys

sys.path.append('.')
from hist_figure import his_figures

if len(sys.argv) > 1:
    prefix = sys.argv[1]
else:
    prefix = time.time()

DATAPATH = '5fold/'
RESULT_PATH = './results/'
feature_num = 25
batch_num = 2
# batch_size = 32
batch_size = 512
SEQ_LENGTH = 20
STATEFUL = False

scaler = None  # tmp, for fit_transform


# id,usage,date,com_date,week,month,year
# com_date,date,id,month,usage,week,year


def get_data(path_to_dataset='df_dh.csv', sequence_length=20, stateful=False, issplit=True):
    fold_index = 1
    ###
    dtypes = {'sub': 'float', 'super': 'float', 'error': 'float', 'com_date': 'int', 'week': 'str', 'month': 'str',
              'year': 'str', 'numbers': 'int', 'log': 'float', 'id': 'str', 'usage': 'float'}
    parse_dates = ['date']
    print(path_to_dataset)
    df = pd.read_csv(DATAPATH + path_to_dataset, header=0, dtype=dtypes, parse_dates=parse_dates, encoding="utf-8")
    # print(path_to_dataset)
    print(df.columns)
    df = df[df['error'] >= 0]

    # df_test = pd.read_csv(DATAPATH+"test"+str(fold_index)+".csv", header = 0,  dtype=dtypes, parse_dates=parse_dates,encoding="utf-8")
    def helper(x):
        split = list(map(int, x.strip('[').strip(']').split(',')))
        d = {}
        for counter, value in enumerate(split):
            k = str(len(split)) + "-" + str(counter)
            d[k] = value
        return d

    # df_train_temp = df_train['week'].apply(helper).apply(pd.Series)
    df_week = df['week'].apply(helper).apply(pd.Series).as_matrix()  # 7
    df_month = df['month'].apply(helper).apply(pd.Series).as_matrix()  # 12
    df_year = df['year'].apply(helper).apply(pd.Series).as_matrix()  # 3

    df_empty = df[['super', 'com_date', 'error', 'numbers']].copy()
    # print(df_empty)
    df_super = df_empty.ix[:, [0]]
    df_com_date = df_empty.ix[:, [1]]
    df_error = df_empty.ix[:, [2]]
    df_numbers = df_empty.ix[:, [3]]

    X_train_ = np.column_stack((df_super, df_com_date, df_numbers, df_week, df_month))
    Y_train_ = df_error.as_matrix()

    ss_x = preprocessing.MaxAbsScaler()
    ss_y = preprocessing.MaxAbsScaler()
    global scaler
    scaler = ss_y
    # ss_x = preprocessing.StandardScaler()
    array_new = ss_x.fit_transform(df_empty.ix[:, [0]])
    df_super = pd.DataFrame(array_new)

    array_new = ss_x.fit_transform(df_empty.ix[:, [1]])
    df_com_date = pd.DataFrame(array_new)

    array_new = ss_x.fit_transform(df_empty.ix[:, [3]])
    df_numbers = pd.DataFrame(array_new)

    array_new = ss_y.fit_transform(df_empty.ix[:, [2]])
    df_error = pd.DataFrame(array_new)

    df_week = ss_x.fit_transform(df_week)
    df_week = pd.DataFrame(df_week)

    df_month = ss_x.fit_transform(df_month)
    df_month = pd.DataFrame(df_month)

    X_train = np.column_stack((df_super, df_com_date, df_numbers, df_week, df_month))
    Y_train = df_error.as_matrix()
    print('Xshape:' + str(X_train.shape))
    print('Yshape:' + str(Y_train.shape))
    y_arr = Y_train.T.tolist()
    # print(y_arr)

    try:
        y_arr = ss_y.inverse_transform(y_arr)
        #draw_error_line(y_arr[0], df)
        #draw_error_bar(y_arr[0])
    except Exception as e:
        print(e)
    if not issplit:
        print('Xshape:' + str(X_train.shape))
        print('Yshape:' + str(Y_train.shape))
        X_train, X_test, Y_train, Y_test = train_test_split(X_train_, Y_train_, test_size=0.1, shuffle=False)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, shuffle=False)
        return X_train, Y_train, X_test, Y_test, X_val, Y_val
    else:
        return split_CV(X_train, Y_train, sequence_length=sequence_length, stateful=False)


import datetime


def get_data_single_user(path_to_dataset='df_dh.csv', sequence_length=20, stateful=False, issplit=True):
    fold_index = 1
    ###
    dtypes = {'sub': 'float', 'super': 'float', 'error': 'float', 'com_date': 'int', 'week': 'str', 'month': 'str',
              'year': 'str', 'numbers': 'int', 'log': 'float', 'id': 'str', 'usage': 'float'}
    parse_dates = ['date']
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + path_to_dataset)
    df = pd.read_csv(DATAPATH + path_to_dataset, header=0, dtype=dtypes, parse_dates=parse_dates, encoding="utf-8")
    # print(path_to_dataset)
    print(df.columns)
    df = df[df['usage'] >= 0]

    # df_test = pd.read_csv(DATAPATH+"test"+str(fold_index)+".csv", header = 0,  dtype=dtypes, parse_dates=parse_dates,encoding="utf-8")
    def helper(x):
        split = list(map(int, x.strip('[').strip(']').split(',')))
        d = {}
        for counter, value in enumerate(split):
            k = str(len(split)) + "-" + str(counter)
            d[k] = value
        return d

    # df_train_temp = df_train['week'].apply(helper).apply(pd.Series)
    df_week = df['week'].apply(helper).apply(pd.Series).as_matrix()  # 7
    df_month = df['month'].apply(helper).apply(pd.Series).as_matrix()  # 12
    df_year = df['year'].apply(helper).apply(pd.Series).as_matrix()  # 3

    df_empty = df[['com_date', 'usage']].copy()
    # print(df_empty)

    df_com_date = df_empty.ix[:, [0]]
    df_usage = df_empty.ix[:, [1]]

    ss_x = preprocessing.MaxAbsScaler()
    ss_y = preprocessing.MaxAbsScaler()
    global scaler
    scaler = ss_y
    # ss_x = preprocessing.StandardScaler()
    array_new = ss_x.fit_transform(df_empty.ix[:, [0]])
    df_com_date = pd.DataFrame(array_new)

    array_new = ss_y.fit_transform(df_empty.ix[:, [1]])
    df_usage = pd.DataFrame(array_new)

    df_week = ss_x.fit_transform(df_week)
    df_week = pd.DataFrame(df_week)

    df_month = ss_x.fit_transform(df_month)
    df_month = pd.DataFrame(df_month)

    X_train = np.column_stack((df_week, df_month))
    Y_train = df_usage.as_matrix()
    print(X_train)
    print(Y_train.shape)
    y_arr = Y_train.T.tolist()
    # print(y_arr)
    print(df)
    y_arr = ss_y.inverse_transform(y_arr)
    draw_error_line(y_arr[0], df)
    draw_error_bar(y_arr[0])
    # try:
    #
    # except Exception as e:
    #     print(e)
    if not issplit:
        return X_train, Y_train
    else:
        return split_CV(X_train, Y_train, sequence_length=sequence_length, stateful=False)


def inverse_xy_transform(scaler, *para):
    temp = []
    for i in para:
        print(i.reshape(-1, 1))
        temp.append(scaler.inverse_transform(i.reshape(-1, 1)))
    return temp


def split_CV(X_train, Y_train, sequence_length=20, stateful=False):
    """return ndarray
    """
    print(X_train)
    print(Y_train.shape[0])
    result_x = []
    result_y = []
    for index in range(len(Y_train) - sequence_length):
        result_x.append(X_train[index: index + sequence_length])
        # result_y.append(Y_train[index: index + sequence_length])
        result_y.append(Y_train[index + sequence_length])
    X_train = np.array(result_x)
    Y_train = np.array(result_y)
    print(X_train.shape)  # (705, 20, 24)
    print(Y_train.shape)  # (705, 1)

    print('##################################################################')
    if stateful == True:
        # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1,shuffle=False)
        cp_X_train = X_train.copy()
        cp_Y_train = Y_train.copy()

        X_train = cp_X_train[:640, ...]
        X_test = cp_X_train[640:, ...]
        Y_train = cp_Y_train[:640, ...]
        Y_test = cp_Y_train[640:, ...]
        print(X_test.shape[0])  #
        print(Y_test.shape[0])  #
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, shuffle=False)
        print('##################################################################')
    if stateful == False:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, shuffle=False)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, shuffle=False)
    # print(X_train.shape)#(705, 20, 24)
    # print(Y_train.shape)#(705, 1)
    # train_x_disorder = X_train.reshape((X_train.shape[0],X_train.shape[1] , feature_num))
    # test_x_disorder = X_test.reshape((X_test.shape[0],X_test.shape[1], feature_num ))
    # X_val = X_val.reshape((X_val.shape[0], X_val.shape[1] , feature_num))
    # print(train_x_disorder.dtype)

    train_y_disorder = Y_train.reshape(-1, 1)
    test_y_disorder = Y_test.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    print(X_train.shape[0])  # (705, 20, 24)
    print(Y_train.shape[0])  # (705, 1)
    print('@' * 40)
    # print(X_test)
    print(train_y_disorder.shape)
    print('@' * 40)
    return [X_train, train_y_disorder, X_test, test_y_disorder, X_val, Y_val]  # ndarray


def LSTM2(X_train):
    model = Sequential()
    # layers = [1, 50, 100, 1]
    layers = [1, 30, 30, 1]
    if STATEFUL == False:
        model.add(LSTM(
            layers[1],
            input_shape=(X_train.shape[1], X_train.shape[2]),
            stateful=STATEFUL,
            return_sequences=True,
            kernel_initializer='he_normal'
            # , kernel_regularizer=l2(0.01)
        ))
    else:
        model.add(LSTM(
            layers[1],
            # input_shape=(X_train.shape[1], X_train.shape[2]),
            batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
            stateful=STATEFUL,
            return_sequences=True,
            kernel_initializer='he_normal'
            # , kernel_regularizer=l2(0.01)
        ))
    # model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        stateful=STATEFUL,
        return_sequences=False,
        kernel_initializer='he_normal'
        # ,kernel_regularizer=l2(0.01)
    ))
    model.add(Dropout(0.2))
    # model.add(Flatten())
    model.add(Dense(
        layers[3]
        , kernel_initializer='he_normal'
        , kernel_regularizer=l2(0.01)
        , activity_regularizer=l1(0.01)
    ))
    model.add(BatchNormalization())
    model.add(Activation("linear"))

    start = time.time()
    sgd = SGD(lr=1e-3, decay=1e-8, momentum=0.9, nesterov=True)
    ada = Adadelta(lr=1e-4, rho=0.95, epsilon=1e-6)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, decay=1e-8)
    adam = Adam(lr=1e-3)
    # model.compile(loss="mse", optimizer=sgd)

    # try:
    #     model.load_weights("./lstm.h5")
    # except Exception as ke:
    #     print(str(ke))
    model.compile(loss="mse", optimizer=adam)
    print("Compilation Time : ", time.time() - start)
    return model


def draw_error_bar(y_array):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    x = list(range(len(y_array)))
    plt.bar(x, y_array, label='error')
    # plt.legend(handles=[line1, line2,line3])
    plt.legend()
    plt.title('error bar')
    # plt.show()

    axes.grid()
    fig.tight_layout()
    fig.savefig(RESULT_PATH + str(batch_size)  + 'bar_error.png', dpi=300)


def draw_error_line(y_array, df):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    x = list(range(len(y_array)))
    plt.plot(x, y_array, label='error')
    x = list(range(len(df['error'])))
    plt.plot(x, df['error'], label='error')
    # plt.legend(handles=[line1, line2,line3])
    plt.legend()
    plt.title('error plot')
    # plt.show()

    axes.grid()
    fig.tight_layout()
    fig.savefig(RESULT_PATH + str(batch_size)  + 'line_error.png', dpi=300)


def draw_scatter(predicted, y_test, X_test, x_train, y_train, data_file):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    x = list(range(len(predicted)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, y_test.T[0], width=width, label='truth', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, predicted, width=width, label='predict', fc='r')

    # plt.legend(handles=[line1, line2,line3])
    plt.legend()
    plt.title('lstm')
    # plt.show()

    axes.grid()
    fig.tight_layout()
    fig.savefig(RESULT_PATH + str(batch_size)  + data_file + str(prefix) + 'bar_lstm.png', dpi=300)

    fig = plt.figure()
    plt.scatter(y_test.T[0], predicted)
    # plt.plot(y_test.T[0], predicted, linewidth =0.3, color='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('truth')
    plt.ylabel('predict')
    # plt.show()
    fig.savefig(RESULT_PATH + str(batch_size)  + data_file + str(prefix) + '_scatter_lstm.png',
                dpi=300)


def draw_line(predicted, y_test, X_test, x_train, y_train, data_file):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    x = list(range(len(predicted)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, y_test.T[0], width=width, label='True', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, predicted, width=width, label='Predicted', fc='r')

    # plt.legend(handles=[line1, line2,line3])
    plt.legend()
    plt.title('lstm')
    # plt.show()

    axes.grid()
    axes = fig.add_subplot(1, 1, 1)
    fig.tight_layout()
    fig.savefig(RESULT_PATH + str(batch_size)  + data_file + str(prefix) + 'bar_lstm.png', dpi=300)

    fig = plt.figure()
    plt.scatter(y_test.T[0], predicted)
    # plt.plot(y_test.T[0], predicted, linewidth =0.3, color='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    # plt.show()
    fig.savefig(RESULT_PATH + str(batch_size)  + data_file + str(prefix) + '_scatter_lstm.png',
                dpi=300)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    plt.plot(x, y_test.T[0], label='True')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.plot(x, predicted, label='Predicted')
    plt.legend()

    axes.grid()
    fig.tight_layout()
    fig.savefig(RESULT_PATH + str(batch_size)  + data_file + str(prefix) + 'line_lstm.png', dpi=300)


def stat_metrics(X_test, y_test, predicted):
    predicted = np.reshape(predicted, y_test.shape[0])
    train_error = np.abs(y_test - predicted)
    mean_error = np.mean(train_error)
    min_error = np.min(train_error)
    max_error = np.max(train_error)
    std_error = np.std(train_error)
    print(predicted)
    print(y_test.T[0])
    print(np.mean(X_test))

    print("#" * 20)
    print(mean_error)
    print(std_error)
    print(max_error)
    print(min_error)
    print("#" * 20)
    print(X_test[:, 1])
    # 0.165861394194
    # ####################
    # 0.238853857898
    # 0.177678269353
    # 0.915951014937
    # 5.2530646691e-0
    pass




def run_regressor(model=LSTM2, sequence_length = SEQ_LENGTH, data=None, data_file='df_dh.csv', isload_model=True, testonly=False):
    epochs = 1000
    path_to_dataset = data_file

    global mses


    if data is None:

        X_train, y_train, X_test, y_test, X_val, Y_val = get_data(sequence_length=sequence_length, stateful=STATEFUL,
                                                                  path_to_dataset=data_file)
    else:
        X_train, y_train, X_test, y_test, X_val, Y_val = data

    if STATEFUL:
        X_test = X_test[:int(X_test.shape[0] / batch_size) * batch_size]
        y_test = y_test[:int(y_test.shape[0] / batch_size) * batch_size]

    estimator = KerasRegressor(build_fn=lambda x=X_train: model(x))

    # if testonly == True:
    #     # predicted = model.predict(X_test, verbose=1,batch_size=batch_size)
    #     prediction = estimator.predict(X_test)

    #     stat_metrics(X_test, y_test, prediction)
    #     draw_scatter(predicted_arr[0], y_test, X_test, X_train, y_train, data_file)
    #     return

    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=40)
    checkpoint = ModelCheckpoint("./lstm.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)
    ################
    hist = estimator.fit(X_train, y_train, validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stopping],
                         epochs=epochs, batch_size=batch_size, verbose=1)

    # prediction = estimator.predict(X_test)
    score = mean_squared_error(y_test, estimator.predict(X_test))
    estimator_score = estimator.score(X_test, y_test)
    print(score)

    mses.append(score)

    prediction = estimator.predict(X_test)
    print(prediction)
    print(X_test)
    print("##############################################")
    # predicted_arr = prediction.T.tolist()
    # print(predicted_arr)
    global scaler
    prediction_, y_test_, y_train_ = inverse_xy_transform(scaler, prediction, y_test, y_train)
    predicted_df = pd.DataFrame(prediction_)
    y_test_df = pd.DataFrame(y_test_)
    # X_test_df = pd.DataFrame(X_test) #columns
    predicted_df.to_csv(DATAPATH + str(prefix) + data_file + str(batch_size) + str(sequence_length) + "predicted_df.csv")
    y_test_df.to_csv(DATAPATH + str(prefix) + data_file + str(batch_size) + str(sequence_length) + "y_test_df.csv")
    # X_test_df.to_csv(DATAPATH+data_file+"X_test_df.csv")
    draw_scatter(prediction, y_test, X_test, X_train, y_train, data_file)
    his_figures(hist)

    draw_line(prediction, y_test, X_test, X_train, y_train, data_file)
    return predicted_df, y_test_df


if __name__ == '__main__':
    # get_data_single_user()
    x = range(5, 121, 5)

    total_mses =[]

    for i in range(1,11):

        mses = []

        for length in x:
            X_train, y_train, X_test, y_test, X_val, Y_val = get_data(sequence_length=length, stateful=STATEFUL)
            run_regressor(sequence_length = length,data=[X_train, y_train, X_test, y_test, X_val, Y_val],
                          data_file='df_dh.csv', isload_model=True)

        total_mses.append(mses)

    print(total_mses)
    np.save(RESULT_PATH + str(prefix) + 'mses.npy', np.asarray(total_mses))






'''
# stock_predict tf
# https://github.com/LouisScorpio/datamining/blob/master/tensorflow-program/rnn/stock_predict/stock_predict_2.py

# boston tf
# https://blog.csdn.net/baixiaozhe/article/details/54410313

########### consume predict keras
# http://www.cnblogs.com/arkenstone/p/5794063.html

# bike number predict keras
# http://resuly.me/2017/08/16/keras-rnn-tutorial/#%E4%BB%BB%E5%8A%A1%E6%8F%8F%E8%BF%B0

# Multivariate Time Series Forecasting with LSTMs in Keras
# https://zhuanlan.zhihu.com/p/28746221
'''
