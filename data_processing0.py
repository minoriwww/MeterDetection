# -*- coding: utf-8 -*-
####
import numpy as np
import pandas as pd
import datetime
from pandas import ExcelWriter
from pandas import ExcelFile

DATA_PATH = "sitaiqu/" # TODO change location
RID = 690100000052 #huayuan:690100001365 donghui:690100000052
UID = 8105360168 #huayuan:8104424663 donghui:8105360168
SMID = 700418018 #huayuan:700420544 donghui:700418018
SN = 1

def super_and_sub(df, SmeterID=SMID):
    df['group'] = np.where(df['meterID'] == SmeterID,'super','sub')
    return df

def KW_process(filename = DATA_PATH+"kilowatt_everyday_2year.xlsx", sheet_name = SN):
    '''for differnet type of xlsx
    # TODO different sheet
    para:
    ---
    sheet_name: index of sheet at xlsx file
    '''

    df = pd.read_excel(filename, sheet_name = sheet_name)

    #df = df.dropna(thresh = 23).drop_duplicates().fillna(method='bfill')
    df.rename(columns={df.columns[0]: "index",
                       df.columns[1]: "redidentsID",
                       df.columns[2]: "userID",
                       df.columns[3]: "meterID",
                       df.columns[4]: "date",
                       df.columns[5]:"usage",
                       }, inplace=True)

    df = df .drop_duplicates(['meterID','date'])
    df = df[df['usage']>=0]

    super_and_sub(df)
    df.to_csv(path_or_buf="sitaiqu/WWWWW.csv", encoding="utf-8")
    #dtypes = {'index': 'int','date': 'str','sub':'int','super':'int'}
    #df_Sum = np.DataFrame('index',"date","sub","super",dtypes)

    sub= df[(df['group']=='sub')]['usage'].groupby(df['date']).sum()
    super= df[df['group']=='super']['usage'].groupby(df['date']).sum()
    df_sub = pd.DataFrame({'date':sub.index, 'sub':sub.values})
    df_super= pd.DataFrame({'date':super.index, 'super':super.values})
    df_sum = pd.merge(df_sub,df_super,on='date')
    df_sum = df_sum.sort_values(by='date')
    df_sum.to_csv(path_or_buf="sitaiqu/Sum.csv", encoding="utf-8")
    # print(df.describe())
    return df_sum

def AV_process(filename, dtypes={}, parse_dates=None):
    '''
    para:
    ---
    dtypes: {'column_name': 'type'...}
            e.g. {'index': 'int', 'redidentsID': 'str', 'userID': 'str', 'meterID': 'str', 'misc': 'str', 'date': 'str', 'sequence': 'str'}
    parse_dates: name of columns converting to datetime format
    '''
    df = pd.read_csv(filename, header = 0, index_col = ['redidentsID', 'userID', 'meterID', 'date', 'sequence'], dtype=dtypes, parse_dates=parse_dates,encoding="utf-8")

    # df.iloc[:,1:df.shape[1]] = df.iloc[:,1:df.shape[1]].applymap(np.float)
    # df = pd.to_numeric(df, errors='coerce')
    df = df.stack(dropna=False).fillna(method='ffill').fillna(method='bfill').unstack()
    df = df.apply(pd.to_numeric, errors='raise')
    # df.values = df.values.astype(float, copy=True)
    # df.iloc[list(range(1,df.shape[1]))] = df.iloc[list(range(1,df.shape[1]))].apply(pd.to_numeric)
    #df = df.infer_objects()
    #print(df)
    return df






def calculate_UIT(df_A, df_V, method='ffill', groupby = ["userID", "date"],multiplier = 1,sequence='A'):
    ''' TODO
    1. multiply DataFrames A and V elementwise, match index and column with ["userID", "date"]
    2. multiply by 15 min
    3. sum
    '''

    #df_V = df_V.sort_values(by='date')
    #df_V.to_csv("Vtest", encoding="utf-8")

    res = df_V.multiply(df_A.reindex(df_V.index, method=method)) # done 1
    res['daily'] = res.apply(lambda x: x.iloc[3:104].sum()*0.00025*multiplier, axis=1)
    #res.rename(columns={res.columns[4]: "sequence"}, inplace=True)
    #res['sequence'].astype('str')
    res = res.xs(sequence,level='sequence')
    res = res.xs(RID, level='redidentsID')
    res = res.xs(UID, level='userID')
    res = res.xs(SMID, level='meterID')
    res = res['daily']
    res = pd.DataFrame({'date':res.index, 'uit':res.values})

    return res

def caculate22(df_A, df_V, method='ffill', groupby = ["userID", "date"],multiplier = 1,sequence='A'):
    res = df_V.multiply(df_A.reindex(df_V.index, method=method))  # done 1
    res['daily'] = res.apply(lambda x: x.iloc[3:104].sum() * 0.00025 * multiplier, axis=1)
    # res.rename(columns={res.columns[4]: "sequence"}, inplace=True)
    # res['sequence'].astype('str')
    # res = res.xs(sequence,level='sequence')
    # res = res.xs(690100001365, level='redidentsID')
    # res = res.xs(8104424663, level='userID')
    # res = res.xs(700420544, level='meterID')
    # res = res['daily']
    # res = pd.DataFrame({'date':res.index, 'uit':res.values})
    res = res.select(lambda x: (x[4] == sequence and x[2] != 700420544))
    res = res.sum(level='date')
    res = res[['daily']]
    res.reset_index(inplace=True)
    return res


def common_process_toCSV(
            filename,
            sheet_name = 1, # huayuanxiaoqu's position
            encoding="utf-8", have_index = True):
    '''change cloumns' name and drop out NaN
    Then generate csv
    '''
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, encoding=encoding)
    else:
        df = pd.read_excel(filename, sheet_name = SN)
    df = df.dropna(thresh = 23).drop_duplicates()
    if not have_index:
        df.reset_index(inplace=True)
    df.rename(columns={ df.columns[0]: "index",
                        df.columns[1]: "redidentsID",
                        df.columns[2]: "userID",
                        df.columns[3]: "meterID",
                        df.columns[4]: "misc", # !different btw A and V
                        df.columns[5]: "date",
                        df.columns[6]: "sequence",
                          }, inplace=True)

    def _inner(s):
        # s = unicode(s)
        if s.find("A")==0:
            return "A"
        elif s.find("B")==0:
            return "B"
        elif s.find("C")==0:
            return "C"
    df['sequence'] = df['sequence'].apply(_inner)
    #print(df)
    return df

def week_map(x):
    str = list([0,0,0,0,0,0,0])
    str[x]=1
    return str

def month_map(x):
    str = list([0,0,0,0,0,0,0,0,0,0,0,0])
    str[int(x)-1]=1
    return str
def year_map(x):
    str = list([0,0,0])
    if(x=='14'):str[0]=1
    if(x=='15'):str[1]=1
    if(x=='16'):str[2]=1
    return str

def date_format(df,base='2016/01/01'):
    df['com_date'] = (pd.to_datetime(df['date']) - pd.to_datetime(base))
    df['com_date'] = df['com_date'].apply(lambda x: x.days)
    df['week'] = pd.to_datetime(df['date']).apply(lambda x: x.weekday()).map(week_map)
    df['month'] = pd.to_datetime(df['date']).apply(lambda x: x.strftime("%m")).map(month_map)
    df['year'] = pd.to_datetime(df['date']).apply(lambda x: x.strftime("%y")).map(year_map)
    return df



if __name__ == '__main__':

    output_prefix = DATA_PATH+"donghuihui" # TODO change sheet name: other resident

    #################### generate CSV ###########################
    df = common_process_toCSV(filename = DATA_PATH+"electriccurrent_hours_2year.xlsx") # would convert to csv

    df.to_csv(path_or_buf = output_prefix+"_A_after.csv", encoding = "utf-8")

    df = common_process_toCSV(filename = DATA_PATH+"voltage_hours_2year.xlsx", sheet_name = SN, have_index=False)
    df.to_csv(path_or_buf = output_prefix+"_V_after.csv", encoding = "utf-8")


    #################### to_numeric data ###########################

    dtypes = {'index': 'int', 'redidentsID': 'str', 'userID': 'str', 'meterID': 'str', 'misc': 'str', 'sequence': 'str'}
    parse_dates = ['date']

    df_A = AV_process(output_prefix+"_A_after.csv", dtypes=dtypes, parse_dates=parse_dates)
    df_V = AV_process(output_prefix+"_V_after.csv", dtypes=dtypes, parse_dates=parse_dates)
    df_A.to_csv(path_or_buf=output_prefix + "_A_after.csv", encoding="utf-8")
    df_V.to_csv(path_or_buf=output_prefix + "_V_after.csv", encoding="utf-8")
    #print(df_A.dtypes)
    #print(df_V.dtypes)

    df = df_A.mul(df_V)
    res = calculate_UIT(df_A, df_V) #TODO
    res.to_csv(path_or_buf=output_prefix + "_P_after.csv", encoding="utf-8")

    df_sum = KW_process()
    df_sum.to_csv(path_or_buf=output_prefix + "_sub_after.csv", encoding="utf-8")
    print(res)

