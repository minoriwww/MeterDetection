import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy import signal
import data_processing0 as dp
import datetime
import math

from scipy.spatial.distance import pdist, squareform

DATA_PATH = "sitaiqu/samples_image2/"
begin = "2014/08/01"
end = "2016/08/01"
ID=1504523749


def rec_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=100
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z


def single_input(id=ID, DATA_PATH = DATA_PATH, begin = begin, end = end,mode=0, df=None):
    filename = dp.DATA_PATH + "kilowatt_everyday_2year.xlsx"

    dtypes = {'index': 'int', 'redidentsID': 'str', 'userID': 'str', 'meterID': 'str', 'misc': 'str', 'sequence': 'str'}
    parse_dates = ['date']



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
        plt.savefig(DATA_PATH+str(id)+"single_input.png",dpi=300)
        res.to_csv(path_or_buf="sitaiqu/samples2/" + str(id) + "single_input.csv",encoding="utf-8", index=False)
    return res


if __name__ == '__main__':
    single_input()
