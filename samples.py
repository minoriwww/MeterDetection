import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_processing0 as dp
import datetime
import math
import single_input_wave as si
import single_bomb_wave as sb
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_PATH = "sitaiqu/"
filename = DATA_PATH + "kilowatt_everyday_2year.xlsx"
rate = 0.5

def random_pick(some_list,probabilities):
	x=random.uniform(0,1)
	cumulative_probability=0.0
	for item,item_probability in zip(some_list,probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability: break
	return item

def op(id,df):
    c=random_pick([0,1],[rate,1-rate])
    print(c,id)
    if c==0 :
        si.single_input(id=id,df=df)
    else:
        sb.single_bomb(id = id,wat=df)

if __name__ == '__main__':
    dtypes = {'index': 'int', 'redidentsID': 'str', 'userID': 'str', 'meterID': 'str', 'misc': 'str', 'sequence': 'str'}
    parse_dates = ['date']
    
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

    #ids['meterID'].apply(op)
    for i in range(0, ids.iloc[:, 0].size):
        op(ids.loc[i,'meterID'],df)

    print(ids)
    ids.to_csv(path_or_buf=DATA_PATH+'aaaa.csv', encoding="utf-8")

