import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import data_processing0 as dp
import input as ip
import datetime
import math
import time
import copy

def bomb(file_path = '5fold/df_dh.csv', data = None):
    d = pd.read_csv(file_path)
    f = pd.read_csv(file_path)
    wat = pd.read_excel("sitaiqu/kilowatt_everyday_2year.xlsx", sheet_name=dp.SN)

    df=d.sample(n=1)
    df.reset_index(inplace=True)



    date = df.loc[0,'date']

    print(date)

    wat.rename(columns={wat.columns[0]: "index",
                           wat.columns[1]: "redidentsID",
                           wat.columns[2]: "userID",
                           wat.columns[3]: "meterID",
                           wat.columns[4]: "date",
                           wat.columns[5]: "usage",
                           }, inplace=True)
    wat = wat.drop_duplicates(['meterID', 'date'])
    wat = wat[wat['usage'] >= 0]
    wat['date']=pd.to_datetime(wat['date'])
    get_id = wat[wat['date']==date]

    x1 = copy.deepcopy(f['super'])
    y1 = copy.deepcopy(f['error'])

    #plt.scatter(x1, y1, color='r',s=10)

    id = dp.SMID
    while id==dp.SMID:
        get_id = get_id.sample(n=1)
        get_id.reset_index(inplace=True)
        id = get_id.loc[0,'meterID']

    new = wat[(wat['date']>=date) & (wat['meterID']==id)]

    # for i in range(0,new.iloc[:,0].size-1):
    #     new.loc[i,'usage']=new.loc[i,'usage']*(1+i/100)
    #     sum+= new.loc[i,'usage']*i/100

    print(new)

    def update(x):
        k = new[new['date'] == x.loc['date']]
        k.reset_index(inplace=True)
        print(k)
        k = k.loc[0, 'usage']
        i = (pd.to_datetime(x.loc['date']) - pd.to_datetime(date)).days
        i = float(i)
        x.loc['sub'] += k * (0.05 * i / math.sqrt((1 + math.pow((i / 15), 2))))
        return x.loc['sub']
        #return 10

    # print(d[['date','sub']])
    # d['sub'] = d[['date','sub']].apply(lambda x: update(x))
    d1=d[d['date']<date]
    d2=d[d['date']>=date]
    d2.reset_index(inplace=True)

    print( d2[['date','sub']].loc[0].loc['date'] )

    for i in range(0,d2.iloc[:,0].size):
        d2.loc[i,'sub']=update(d2[['date','sub']].iloc[i])



    d=d1.append(d2)

    d['error'] = d['super'] - d['sub']

    print(f)
    print(d)

    x2 = d['super']
    y2 = d['error']

    # print(x1.shape)
    # print(x2.shape)

    y = list(map(lambda x: x[0] - x[1], zip(y1.tolist(), y2.tolist())))

    # print(len(y1.tolist()))
    # print(len(y2.tolist()))
    # print(len(y))



    d=d.drop(columns=['index','sub'])




    plt.scatter(x2,y,marker='x',s=10)
    plt.legend()
    plt.savefig('sitaiqu/compare.png',dpi=300)


    d.to_csv(path_or_buf="sitaiqu/bomb_test/"+str(time.time())+str(date)+"bomb_test.csv", encoding="utf-8", index=False)
    return d

if __name__ == '__main__':
    bomb()
