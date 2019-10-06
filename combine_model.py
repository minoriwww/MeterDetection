# -*- coding: utf-8 -*-
#tested in py2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth=True
tf.Session(config=config)
import numpy as np
np.random.seed(1337)



import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib as mpl
import random
from imutils import paths
mpl.use('Agg')


import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
plt.switch_backend('agg')
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras import regularizers
from keras import backend as K
from keras.models import Model, Sequential
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD, Adadelta, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution1D, Convolution2D, LSTM, Reshape, TimeDistributed, Bidirectional
import keras.layers.convolutional as conv
from keras.regularizers import l1, l2
import sklearn.metrics as m
import time
from scipy import interp
import cv2
from sklearn.model_selection import StratifiedKFold
# from DProcess import convertRawToXY
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import hyperopt as hp
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials
from keras.utils import plot_model

from ROC_PR_threshold import roc_curve
import sys

seed = 7
np.random.seed(seed)

PRECODE="biLSTM"


if len(sys.argv) > 1:
    DATAPATH = sys.argv[1]
else:
    DATAPATH = 'sitaiqu/'

def helper(x):
    """old version of date format helper, run in py2 and old pandas
    """
    splited_list  = list(map(int, x.strip('[').strip(']').split(',')))

    d = {}
    for counter, value  in enumerate(splited_list):
        k = str(len(list(splited_list)))+"-"+str(counter)
        d[k] = int(value)

    return d
def date_helper(x, column_name):
    return pd.Series(list(map(int, x[column_name].strip('[').strip(']').split(','))))

def png_folder_processing(path=DATAPATH+'samples_image1/',seed=42):

    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(seed)
    #random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (128, 128))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        if imagePath.split('.')[0].endswith('bomb'):
            # y_array.append(1)
            labels.append(1)
            print('1')
        else:
            # y_array.append(0)
            labels.append(0)
            print('0')

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors

    return data,labels



def csv_folder_processing(DATAPATH=DATAPATH+'samples1/'):

    X_array = []
    y_array = []
    line_nb = 0
    col_num = 0
    num_users = 1
    for lists in os.listdir(DATAPATH):
        sub_path = os.path.join(DATAPATH, lists)
        # print(sub_path)
        if os.path.isfile(sub_path):
            num_users += 1

    X = np.zeros((num_users, 729, 23), dtype=np.float32)
    y = np.zeros(num_users, dtype=np.float32)

    g = os.walk(DATAPATH)
    for path,dir_list,file_list in g:
        for j, file_name in enumerate(file_list, 1):

            print(file_name)
            if not file_name.startswith('.'): X_csv = csv_processing(csv_file_name = os.path.join(path, file_name), padding_line = 729)
            X[j] = X_csv[:729, ...]
            #
            if file_name.split('.')[0].endswith('bomb'):
                y[j] = 1
                print('1')
            else:
                y[j] = 0
                print('0')

    return X

def csv_processing(csv_file_name = "", padding_line = 729):


    fold_index = 1
    ###
    dtypes = {'sub': 'float', 'super': 'float', 'error': 'float', 'com_date': 'int', 'week': 'str', 'month': 'str', 'year': 'str', 'numbers':'int', 'log':'float', 'id': 'str', 'usage': 'float'}
    # , 'A_mean': 'float', 'V_mean': 'float'}
    parse_dates = ['date']
    print("start "+ csv_file_name)
    df = pd.read_csv(csv_file_name, header = 0,  dtype=dtypes, parse_dates=parse_dates,encoding="utf-8")
    # df_test = pd.read_csv(DATAPATH+"test"+str(fold_index)+".csv", header = 0,  dtype=dtypes, parse_dates=parse_dates,encoding="utf-8")
    df = df.sample(frac=1)

    df_week = df.apply(lambda x: date_helper(x,'week'), axis=1)
    df_month = df.apply(lambda x: date_helper(x,'month'), axis=1)
    df_year = df.apply(lambda x: date_helper(x,'year'), axis=1)


    # df_week = df['week'].apply(helper, axis=1).values #7
    # df_month = df['month'].apply(helper, axis=1).values #12
    # df_year = df['year'].apply(helper, axis=1).values #3
    print("df_week")
    print(df_week)
    '''
    X_train = df[['super','com_date']].as_matrix()
    X_train = np.column_stack((X_train, df_week, df_month, df_year))

    Y_train = df[['error']].as_matrix()

    '''
    df_empty = df[[ 'usage', 'com_date']].copy()
    # print(df_empty)

    # ss_x = preprocessing.MaxAbsScaler()
    ss_x = preprocessing.StandardScaler()
    array_new = ss_x.fit_transform(df_empty.iloc[:,[0]])
    df_usage = pd.DataFrame(array_new)

    array_new = ss_x.fit_transform(df_empty.iloc[:,[1]])
    df_com_date = pd.DataFrame(array_new)

    df_week = ss_x.fit_transform(df_week)
    # df_week = pd.DataFrame(df_week)

    df_month = ss_x.fit_transform(df_month)
    # df_month = pd.DataFrame(df_month)

    df_year = ss_x.fit_transform(df_year)
    # df_year = pd.DataFrame(df_year)
    # X_train = df_empty.ix[:,[2]].as_matrix()

    print(df_usage.shape,df_week.shape,df_month.shape,df_year.shape)
    X_train = np.column_stack((df_usage,  df_week, df_month, df_year)) #+ 1 7 12 3 = 23
    if df.shape[0]<=padding_line:
        X_train = np.pad(X_train, ((0, padding_line), (0,0)), 'edge')

    print(X_train)
    return X_train


def Combine_model(X1_train,X2_train):

    VGGmodel = keras.applications.vgg16.VGG16(include_top=False
                                              , weights='imagenet'
                                              # , input_tensor=inputs
                                              , input_shape=X1_train.shape[1:]
                                              , pooling=None
                                              # , classes=1000
                                              )

    VGGupModel = keras.applications.vgg16.VGG16(include_top=False
                                              , weights=None
                                              # , input_tensor=inputs
                                              , input_shape=X1_train.shape[1:]
                                              , pooling=None
                                              # , classes=1000
                                              )


    ResNetmodel = keras.applications.resnet.ResNet50(include_top=False

                                              , weights='imagenet'
                                              # , input_tensor=inputs
                                              , input_shape=X1_train.shape[1:]
                                              , pooling=None
						)
    x1= Flatten()(ResNetmodel.output)
    x1= Dropout(0.5)(x1)
    x1= Dense(128)(x1)

    xs = Input(X2_train.shape[1:])
    '''
    x_image = conv.Convolution1D(16, 5, padding='same', init='he_normal', W_regularizer=l1(0.01))(xs)
    x_image = BatchNormalization()(x_image)
    x_image = Activation('relu')(x_image)

    x_image = conv.Convolution1D(64, 5, padding='same', init='he_normal', W_regularizer=l1(0.01))(x_image)
    x_image = Activation('relu')(x_image)
   
    x_image = Flatten()(x_image)
    x_image = Dense(128, init='he_normal', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01))(x_image)
    '''
    # BiLstm
    blmodel = Sequential()
    input_shape = X2_train.shape[1:]
    blmodel.add(Bidirectional(LSTM(units=20, return_sequences=True), input_shape=input_shape))
    blmodel.add(Flatten())
    blmodel.add(Dropout(0.5))
    #blmodel.add(BatchNormalization())

    #blmodel.add(TimeDistributed())

    blmodel.add(Dense(128, activation='sigmoid'))

    #x_image = keras.layers.Concatenate(axis=-1)([x1, x_image])
    x_image = keras.layers.Concatenate(axis=-1)([x1,blmodel.output])

    # x_image = keras.layers.Add()([x1, x_image])
    x_image = Dropout(0.5)(x_image)
    preds = Dense(2, init='he_normal', activation='sigmoid')(x_image)

    ada = Adadelta(lr=1e-3, rho=0.95, epsilon=1e-6)

    model = Model([ResNetmodel.input,blmodel.input], preds)

    # Compile model
    model.summary()
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])

    #categorical_crossentropy
    #binary_crossentropy

    # history = model.fit(X_train, Y_train,
    #                     batch_size = 20,
    #                     epochs = 50,
    #                     verbose = 1,
    #                     validation_data = (X_test, Y_test))
    # score = model.evaluate(X_test, Y_test, verbose=1)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    return model



def kfold_evaluation_plot(X1, X2,y):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    recalls = []
    precisions = []
    praucs = []
    mean_precision = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    plt.figure(figsize=(16, 6))

    kf = KFold(n_splits=5)

    early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=30)

    lw = 2
    i = 1

    for train_index, test_index in kf.split(X1):
        results_filename = str(time.time())
        checkpoint = ModelCheckpoint("./weights/dnn_weights" + results_filename + ".h5", monitor='val_loss', verbose=1,
                                     save_best_only=True)
        X1_train, X1_test = X1[train_index], X1[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        X1_val, Y_val = X1_test, Y_test

        X2_train, X2_test = X2[train_index], X2[test_index]
        X2_val = X2_test,

        Y_test = convert_y(Y_test)

        print(Y_test)
        Y_train= convert_y(Y_train)

        model = Combine_model(X1_train,X2_train)
        model.fit([X1_train,X2_train], Y_train, batch_size=1, epochs=1000,
                 validation_data=([X1_test,X2_test], Y_test),
                 callbacks=[early_stopping, checkpoint])

        prediction = model.predict([X1_test,X2_test])

        print(prediction)

        fpr, tpr, thresholds = m.roc_curve(Y_test.T[1], prediction.T[1], pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = m.auc(fpr, tpr)
        aucs.append(roc_auc)

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


        plt.subplot(1,2,2)
        precision, recall, th = m.precision_recall_curve(Y_test.T[1], prediction.T[1])
        recalls.append(interp(mean_precision, precision, recall))

        recall = recall[::-1]
        precision = precision[::-1]

        precisions.append(interp(mean_recall, recall, precision))

        prc_auc = m.auc(recall, precision)
        praucs.append(prc_auc)

        plt.plot(recall, precision, lw=1, alpha=0.3, label='PRC fold %d (AUC = %0.2f)' % (i, prc_auc))

        i += 1
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = m.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)

    mean_precision = np.mean(precisions, axis=0)
    mean_prauc = m.auc(mean_recall, mean_precision)
    std_prauc = np.std(praucs)
    # mean_recall = np.mean(recalls,axis=0)



    plt.plot(mean_recall, mean_precision, color='b',
             label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_prauc, std_prauc),
             lw=2, alpha=.8)

    std_precision = np.std(np.array(precisions),axis=0)

    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)

    plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.legend(loc="lower left")


    plt.savefig("sitaiqu/classify_figures/%.2f roc_curve_user_classify_" % mean_auc + results_filename + PRECODE + ".png",dpi=300)


def convert_y(y):
    y_dim2=[]
    for i in y:
        if (i==0): y_dim2.append([1,0])
        if (i==1): y_dim2.append([0,1])

    return np.array(y_dim2)



if __name__ == '__main__':
    X1,y = png_folder_processing()
    X2 = csv_folder_processing()
    kfold_evaluation_plot(X1,X2,y)
