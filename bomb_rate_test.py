
import os
dirs = "sitaiqu/rates_test/"
if not os.path.exists(dirs):
    os.makedirs(dirs)

import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import data_processing0 as dp
import datetime
import math
import random
from scipy.spatial.distance import pdist, squareform

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution1D, Convolution2D, LSTM, Reshape
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

def random_pick(some_list,probabilities):
	x=random.uniform(0,1)
	cumulative_probability=0.0
	for item,item_probability in zip(some_list,probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability: break
	return item

def op(id,df,rate,file_path):
    c=random_pick([0,1],[rate,1-rate])
    print(c,id)
    if c==0 :
        single_input(id=id,df=df,DATA_PATH=file_path)
    else:
        single_bomb(id = id,wat=df,DATA_PATH=file_path)

def rec_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=100
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z


def single_bomb(id,DATA_PATH = "", date = "2015/05/01",wat=None):

    if not os.path.exists(DATA_PATH+'csv'):
        os.makedirs(DATA_PATH+'csv')
    if not os.path.exists(DATA_PATH+'png'):
        os.makedirs(DATA_PATH+'png')

    d = single_input(id=id,mode=1,df=wat)


    df = d.sample(n=1)
    df.reset_index(inplace=True)
    date = df.loc[0, 'date']

    id = id


    new = wat[(wat['date']>=date) & (wat['meterID']==id)]


    #print(new)

    def update(x):
        i=(pd.to_datetime(x.loc['date'])-pd.to_datetime(date)).days
        x.loc['usage']=x.loc['usage']*(1+i/100)
        #i = float(i)
        #x.loc['usage'] += x.loc['usage'] * (0.05 * i / math.sqrt((1 + math.pow((i / 15), 2))))
        return x.loc['usage']


    d1=d[d['date']<date]
    d2=d[d['date']>=date]
    d2.reset_index(inplace=True)

    for i in range(0,d2.iloc[:,0].size-1):
        d2.loc[i,'usage']=update(d2[['date','usage']].iloc[i])

    d=d1.append(d2)
    d=d.drop(columns=['index'])

    d=d[['id','usage','date','com_date','week','month','year']]

    x=d['usage']
    rec = rec_plot(x)
    plt.imshow(rec)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(DATA_PATH+'png/' + str(id) + "single_bomb.png",dpi=300)
    d.to_csv(path_or_buf=DATA_PATH+'csv/' + str(id) + "single_bomb.csv", encoding="utf-8",index=False)


def single_input(id=123, DATA_PATH="", mode=0, df=None):
    if not os.path.exists(DATA_PATH+'csv'):
        os.makedirs(DATA_PATH+'csv')
    if not os.path.exists(DATA_PATH+'png'):
        os.makedirs(DATA_PATH+'png')

    df = df[df['meterID'] == id]

    res = pd.DataFrame()
    res['id'] = df['meterID']
    res['usage'] = df['usage']
    res['date'] = pd.to_datetime(df['date'])
    res = dp.date_format(res, base='2014/8/3')
    res = res.sort_values(by='date')
    # df = df[(df['date'] >= pd.to_datetime(begin)) & (df['date'] <= pd.to_datetime(end))]
    x = res['usage']


    rec = rec_plot(x)
    plt.imshow(rec)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if mode ==0 :

        plt.savefig(DATA_PATH+'png/'+str(id)+"single_input.png",dpi=300)
        res.to_csv(path_or_buf=DATA_PATH+'csv/'+ str(id) + "single_input.csv",encoding="utf-8", index=False)
    return res

def generate_sample(filepath,rate = 0.5,filename="sitaiqu/kilowatt_everyday_2year.xlsx"):

    df = pd.read_excel(filename, sheet_name=dp.SN)
    df.rename(columns={df.columns[0]: "index",
                       df.columns[1]: "redidentsID",
                       df.columns[2]: "userID",
                       df.columns[3]: "meterID",
                       df.columns[4]: "date",
                       df.columns[5]: "usage",
                       }, inplace=True)
    df = df[df['meterID'] != dp.SMID]
    df = df.drop_duplicates(['meterID', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['usage'] >= 0]

    df = df.sort_values(by='date')

    ids = df['meterID'].to_frame()
    ids = ids.drop_duplicates().reset_index()

    # ids['meterID'].apply(op)
    for i in range(0, ids.iloc[:, 0].size):
        op(ids.loc[i, 'meterID'], df,rate,filepath)
def helper(x):
    split  = map(int, x.strip('[').strip(']').split(','))
    d = {}
    for counter, value  in enumerate(split):
        k = str(len(list(split)))+"-"+str(counter)
        d[k] = int(value)

    return d

def png_folder_processing(path,seed=42):

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
        if str.find(imagePath,'bomb')!=-1:
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



def csv_folder_processing(DATAPATH='samples2/'):

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

def date_helper(x, column_name):
    return pd.Series(list(map(int, x[column_name].strip('[').strip(']').split(','))))

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
    # df_train_temp = df_train['week'].apply(helper).apply(pd.Series)
    # df_week = df['week'].apply(helper).apply(pd.Series).as_matrix() #7
    # df_month = df['month'].apply(helper).apply(pd.Series).as_matrix() #12
    # df_year = df['year'].apply(helper).apply(pd.Series).as_matrix() #3

    df_week = df.apply(lambda x: date_helper(x, 'week'), axis=1)
    df_month = df.apply(lambda x: date_helper(x, 'month'), axis=1)
    df_year = df.apply(lambda x: date_helper(x, 'year'), axis=1)

    '''
    X_train = df[['super','com_date']].as_matrix()
    X_train = np.column_stack((X_train, df_week, df_month, df_year))

    Y_train = df[['error']].as_matrix()

    '''
    df_empty = df[[ 'usage', 'com_date']].copy()
    # print(df_empty)

    # ss_x = preprocessing.MaxAbsScaler()
    ss_x = preprocessing.StandardScaler()
    array_new = ss_x.fit_transform(df_empty.ix[:,[0]])
    df_usage = pd.DataFrame(array_new)

    array_new = ss_x.fit_transform(df_empty.ix[:,[1]])
    df_com_date = pd.DataFrame(array_new)

    df_week = ss_x.fit_transform(df_week)
    # df_week = pd.DataFrame(df_week)

    df_month = ss_x.fit_transform(df_month)
    # df_month = pd.DataFrame(df_month)

    df_year = ss_x.fit_transform(df_year)
    # df_year = pd.DataFrame(df_year)
    # X_train = df_empty.ix[:,[2]].as_matrix()
    X_train = np.column_stack((df_usage,  df_week, df_month, df_year)) #+ 1 7 12 3 = 23
    if df.shape[0]<=padding_line:
        X_train = np.pad(X_train, ((0, padding_line), (0,0)), 'edge')


    return X_train


def Combine_model(X1_train,X2_train):

    VGGmodel = keras.applications.vgg16.VGG16(include_top=False
                                              , weights='imagenet'
                                              # , input_tensor=inputs
                                              , input_shape=X1_train.shape[1:]
                                              , pooling=None
                                              # , classes=1000
                                              )
    x1= Flatten()(VGGmodel.output)
    x1= Dropout(0.5)(x1)
    x1= Dense(128)(x1)

    xs = Input(X2_train.shape[1:])
    x_image = conv.Convolution1D(16, 5, padding='same', init='he_normal', W_regularizer=l1(0.01))(xs)
    x_image = BatchNormalization()(x_image)
    x_image = Activation('relu')(x_image)

    x_image = conv.Convolution1D(64, 5, padding='same', init='he_normal', W_regularizer=l1(0.01))(x_image)
    x_image = Activation('relu')(x_image)
    # x_image = BatchNormalization()(x_image)

    # x_image = conv.Convolution2D(128, (2, 2), padding='same',init='he_normal')(x_image)
    # x_image = Activation('relu')(x_image)
    # x_image = BatchNormalization()(x_image)
    x_image = Flatten()(x_image)
    x_image = Dense(128, init='he_normal', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01))(x_image)

    x_image = keras.layers.Add()([x1, x_image])
    x_image = Dropout(0.5)(x_image)
    preds = Dense(2, init='he_normal', activation='sigmoid')(x_image)

    ada = Adadelta(lr=1e-3, rho=0.95, epsilon=1e-6)

    model = Model([VGGmodel.input,xs], preds)

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

    # print("Y:")
    # print(y)

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

        # print(Y_test)
        Y_train= convert_y(Y_train)

        model = Combine_model(X1_train,X2_train)
        model.fit([X1_train,X2_train], Y_train, batch_size=1, epochs=1000,
                 validation_data=([X1_test,X2_test], Y_test),
                 callbacks=[early_stopping, checkpoint])

        prediction = model.predict([X1_test,X2_test])

        # print(prediction)

        fpr, tpr, thresholds = m.roc_curve(Y_test.T[1], prediction.T[1], pos_label=1)

        # print(tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = m.auc(fpr, tpr)
        aucs.append(roc_auc)

        i += 1
    # plt.subplot(1, 2, 1)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Chance', alpha=.8)


    # print(tprs)
    mean_tpr = np.mean(tprs, axis=0)

    # print (mean_fpr, mean_tpr)
    mean_tpr[-1] = 1.0
    mean_auc = m.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # print (mean_auc,std_auc)
    return mean_auc, std_auc


def convert_y(y):
    y_dim2=[]
    for i in y:
        if (i==0): y_dim2.append([1,0])
        if (i==1): y_dim2.append([0,1])

    return np.array(y_dim2)



def run_cnn(path):
    X1, y = png_folder_processing(path = path+'png/')
    X2 = csv_folder_processing(DATAPATH = path+'csv/')
    mean_auc, std_auc = kfold_evaluation_plot(X1, X2, y)

    # print (mean_auc, std_auc)
    return mean_auc, std_auc


if __name__ == '__main__':
    # rate = 0.9
    # while rate<0.96:
    #     file_path = dirs+str(rate)+'/'
    #     if not os.path.exists(file_path):
    #         os.makedirs(file_path)
    #         generate_sample(file_path,rate)
    #     rate+=0.01

    auc_x=[]
    auc_y=[]
    stds = []

    rate = 0.9

    while rate<0.96:
        print ("Now working with rate ="+str(rate)[:4])
        file_path = dirs+str(rate)[:4]+'/'
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        #generate_sample(file_path,rate)
        aucc, std = run_cnn(file_path)


        print ("************************************************")
        print (aucc, std)
        auc_x.append(rate)
        auc_y.append(aucc)
        stds.append(std)
        rate += 0.01

    print(auc_x)
    print(auc_y)

    upper = []
    lower = []
    for i in range(len(auc_y)):
        upper.append(auc_y[i] + stds[i])
        lower.append(auc_y[i] - stds[i])

    plt.plot(auc_x,auc_y)

    plt.fill_between(auc_x, lower, upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Rate')
    plt.ylabel('Area Under Curve')

    plt.savefig("sitaiqu/rates_test/rate"+str(time.time())+".png", dpi=300)

