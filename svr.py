# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np  # numpy库
import pandas
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from mylstm import get_data

filename='5fold/df_dh.csv'

def helper(x):
    split  = map(int, x.strip('[').strip(']').split(','))
    d = {}
    for counter, value  in enumerate(split):
        k = str(len(list(split)))+"-"+str(counter)
        d[k] = int(value)

    return d


df=pd.read_csv(filename)
df_week = df['week'].apply(helper).apply(pd.Series).as_matrix() #7
df_month = df['month'].apply(helper).apply(pd.Series).as_matrix() #12
df_year = df['year'].apply(helper).apply(pd.Series).as_matrix() #3

X=np.column_stack((df.drop(['date','week','month','year','error','sub'],axis=1),  df_week, df_month, df_year))
y=df['error']

X_train = X.copy()
Y_train = y.copy()

#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1,shuffle=False)
X_train, Y_train, X_test, Y_test, X_val, Y_val = get_data(issplit = False)
print('Xshape:'+str(X_train.shape))
print('Yshape:'+str(Y_train.shape))
print(X)


# 训练回归模型
n_folds = 5  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
#model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['ElasticNet', 'GBR']  # 不同模型的名称列表
model_dic = [ model_etc, model_gbr]  # 不同回归模型对象的集合
cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, X_train, Y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(X_train, Y_train).predict(X_test))  # 将回归训练中得到的预测y存入列表
# 模型效果指标评估
n_samples, n_features = X_test.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(2):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(Y_test, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print (70 * '-')  # 打印分隔线
print ('cross validation result:')  # 打印输出标题
print (df1)  # 打印输出交叉检验的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('short name \t full name')  # 打印输出缩写和全名标题
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  # 打印分隔线

# 模型效果可视化

lstm=pd.read_csv('5fold/1547574783.74df_dh.csv51220predicted_df.csv')
lstm_test=pd.read_csv('5fold/1547574783.74df_dh.csv51220y_test_df.csv')
lstm=lstm.drop('Unnamed: 0', axis=1)
lstm_test=lstm_test.drop('Unnamed: 0', axis=1)
#print(lstm)

plt.switch_backend('agg')
plt.figure(figsize=(8,4))  # 创建画布
#print(X_test.shape[0])
lw=2
plt.plot(np.arange(X_test.shape[0]), Y_test, lw=lw,color='r', label='True values')  # 画出原始值的曲线
t=8

k=Y_test.reshape(1,-1).flatten()
#print(Y_test)

pp=[i+t for i in k]
pm=[i-t for i in k]

pm = [(i+abs(i))/2 for i in pm]
#print(pp)
#print(pm)

plt.fill_between(np.arange(Y_test.shape[0]),pm , pp, facecolor='pink', interpolate=True, alpha=0.5)
color_list = ['g', 'c', 'g', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', '*']  # 样式列表

ndir=[0,0,0]
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    pre_y_list[i]=(pre_y_list[i]+abs(pre_y_list[i]))/2
    for j,x in enumerate(pre_y_list[i]):
        print(j,x)
        if (x<=pp[j]) and (x>=pm[j]): ndir[i]+=1
    plt.plot(np.arange(X_test.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线

lstm = (lstm + abs(lstm))/2
plt.plot(np.arange(2,lstm.shape[0]+2), lstm,'b',label='LSTM')

print(lstm)
lstm=lstm.values.tolist()
for i,x in enumerate(lstm):
    print(i,x)
    if (x <= pp[i]) and (x >=pm[i]): ndir[2] += 1

# plt.title('regression result comparison')  # 标题
plt.legend(loc='upper left')  # 图例位置
plt.ylabel('Predicted E')  # y轴标题
plt.xlabel('Days')
results_filename = str(time.time())
plt.savefig("svr/svr"+"_"+results_filename+".png",dpi=300)
print('number of days in range:')
print(ndir)