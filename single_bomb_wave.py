import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_processing0 as dp
from scipy import signal
import input as ip
import datetime
import math
import single_input_wave as si
from scipy.spatial.distance import pdist, squareform
DATE='2015/08/01'
DATA_PATH = "sitaiqu/samples_image/"
import os
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.isdir("sitaiqu/samples/"):
    os.makedirs("sitaiqu/samples/")


filename = dp.DATA_PATH+'/single_input.csv'

def rec_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=100
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def single_bomb(id,filename = filename, date = DATE,wat=None):

    d = si.single_input(id=id,mode=1,df=wat)


    df = d.sample(n=1)
    df.reset_index(inplace=True)
    date = df.loc[0, 'date']

    id = id


    new = wat[(wat['date']>=date) & (wat['meterID']==id)]


    #print(new)

    def update(x):
        i=(pd.to_datetime(x.loc['date'])-pd.to_datetime(date)).days
        #x.loc['usage']=x.loc['usage']*(1+i/100)
        i = float(i)
        x.loc['usage'] += x.loc['usage'] * (0.05 * i / math.sqrt((1 + math.pow((i / 15), 2))))
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
    plt.savefig(DATA_PATH + str(id) + "single_bomb.png",dpi=300)
    d.to_csv(path_or_buf=dp.DATA_PATH + "/samples/" + str(id) + "single_bomb.csv", encoding="utf-8",index=False)

if __name__ == '__main__':
    single_bomb(id=1504523749)
