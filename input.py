# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_processing0 as dp
import datetime

# def get_mean(df,df_A,df_V):
#     df['A_mean']=df_A.apply(lambda x: x.iloc[3:104].mean(), axis=1)
#     df['V_mean']=df_V.apply(lambda x: x.iloc[3:104].mean(), axis=1)
#     return df

if __name__ == '__main__':
    DATA_PATH = "./sitaiqu/"

    output_prefix = DATA_PATH + "huayuanxiaoqu"
    begin = "2016/03/01"
    end = "2016/08/01"
    dtypes = {'index': 'int', 'redidentsID': 'str', 'userID': 'str', 'meterID': 'str', 'misc': 'str', 'sequence': 'str'}
    parse_dates = ['date']
    df_A = dp.AV_process(output_prefix + "_A_after.csv", dtypes=dtypes, parse_dates=parse_dates)
    df_A['A_mean'] = df_A.apply(lambda x: x.iloc[3:104].mean(), axis=1)
    A_mean=df_A['A_mean'].to_frame()
    df_V = dp.AV_process(output_prefix + "_V_after.csv", dtypes=dtypes, parse_dates=parse_dates)
    df_V['V_mean'] = df_V.apply(lambda x: x.iloc[3:104].mean(), axis=1)
    V_mean = df_V['V_mean'].to_frame()
    df =dp.KW_process()
    df = df.sort_values(by='date')
    df['error']=df['super']-df['sub']
    df=dp.date_format(df,base='2014/8/3')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['error'] >= 0]
    df = df.sort_values(by='date')
    #df = df[(df['date'] >= pd.to_datetime(begin)) & (df['date'] <= pd.to_datetime(end))]

    A_mean = A_mean.xs('A', level='sequence')
    A_mean = A_mean.xs(690100001365, level='redidentsID')
    A_mean = A_mean.xs(8104424663, level='userID')
    A_mean = A_mean.xs(700420544, level='meterID')
    V_mean = V_mean.xs('A', level='sequence')
    V_mean = V_mean.xs(690100001365, level='redidentsID')
    V_mean = V_mean.xs(8104424663, level='userID')
    V_mean = V_mean.xs(700420544, level='meterID')

    A_mean.reset_index(inplace=True)
    V_mean.reset_index(inplace=True)
    mean=pd.merge(A_mean,V_mean)
    mean = mean.sort_values(by='date')
    mean['date'] = pd.to_datetime(mean['date'])
    mean = mean[(mean['date'] >= pd.to_datetime(begin)) & (mean['date'] <= pd.to_datetime(end))]
    res=pd.merge(df,mean)
    res = res.sort_values(by='date')
    res = res[res['error']>=0]
    df.to_csv(path_or_buf=DATA_PATH+"df.csv", encoding="utf-8",index=False)
    mean.to_csv(path_or_buf=DATA_PATH+"mean.csv", encoding="utf-8",index=False)
    res.to_csv(path_or_buf=DATA_PATH+"input.csv", encoding="utf-8",index=False)

