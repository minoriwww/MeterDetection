#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# predicted_filename = '/Users/sunguangyu/Downloads/电表/1528641961.4759896df_dh.csv51220predicted_df.csv'
# test_filename = '/Users/sunguangyu/Downloads/电表/1528641961.4759896df_dh.csv51220y_test_df.csv'

predicted_filename = '/Users/sunguangyu/Downloads/电表/1536784794.03df_dh.csv51220predicted_df.csv'
test_filename = '/Users/sunguangyu/Downloads/电表/1536784794.03df_dh.csv51220y_test_df.csv'

l=4
t=0.5
def check(l=l, t=t, predicted_filename = predicted_filename, test_filename = test_filename):
    flag=False

    predicted = pd.read_csv(predicted_filename)
    test = pd.read_csv(test_filename)

    error = predicted - test
    #print(error)
    #error.reset_index(inplace=True)

    error.rename(columns={error.columns[0]: "index", error.columns[1]: "data", }, inplace=True)
    error['data']=error['data'].apply(lambda x: abs(x))
    a = [0 for x in range(-1, 1000)]

    #print(error)
    def get_it(p,x):
        x=round(x)
        return p.iloc[x,0]

    for i in range(0, error.iloc[:, 0].size-1-l):
        for j in range(0, l-1):
            if error.loc[i+j, 'data'] < t:
                #print(i,i+j,error.loc[i+j, 'data'])
                continue
            else:
                a[i+j]=a[i+j-1]+1
                if a[i+j] == l:
                    print(i+j)
                    flag=True
        if flag:
            break
    #print(a)
    test.rename(columns={test.columns[0]: "index", test.columns[1]: "data", }, inplace=True)
    test['data']=test['data'].apply(lambda x: abs(x))

    predicted=predicted.drop('Unnamed: 0', axis=1)
    predicted=predicted['0'].tolist()
    test = test.drop('index', axis=1)
    test = test['data'].tolist()
    print(predicted)
    print(error)

    pp=[i+t for i in predicted]
    pm=[i-t for i in predicted]

    # for i in range(0,71):
    #     print(get_it(predicted,i))
    #     if (get_it(error,i) > get_it(predicted,i)+t): plt.fill_between(np.linspace(i,i+1),get_it(error,i),get_it(predicted,i)+t,facecolor='purple')
    x=np.arange(0,72)
    y1=pp
    y2=test

    # plt.fill_between(x, y1, y2, where= (y1 > y2), facecolor='green',interpolate=True, alpha=1.5)
    # plt.fill_between(x, y1, 0, facecolor='white', interpolate=True, alpha=1)

    plt.fill_between(x, pm, test, where=(pm < test), facecolor='green', interpolate=True, alpha=1.5)
    plt.fill_between(x, test, -1, facecolor='white', interpolate=True, alpha=1)

    plt.plot(pp,'r--',label='Upper Bound')
    plt.plot(pm,'g--',label='Lower Bound')
    plt.plot(test,'b',label='Test(malfunction)')
    plt.legend()
    plt.xlim(0,71)

    plt.show()


if __name__ == '__main__':
    check()