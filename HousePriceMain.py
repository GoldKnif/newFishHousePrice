import GetHouseData
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

train_file = "kaggle_house_train"
test_file = "kaggle_house_test"

#1.读取数据
train_data = pd.read_csv("data/kaggle_house_pred_train.csv")
test_data = pd.read_csv("data/kaggle_house_pred_test.csv")

# print(test_data.shape)
# print(test_data.shape)
# 可以读取导数据 ok 12/09

#print(train_data.iloc[0:4,[0,1,2,3,-3,-22,-1]])
    #数据检查成功 ok 12/09


#2.训练集和测试集进行拼接
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
# print(all_features.shape)
# print(all_features.head())
    #测试数据拼接输出成功 ok 12/09

# 3.处理数据，对数据进行归一化 赋予缺失均值，离散变量转独热编码
numeric_features = all_features.dtypes[all_features.dtypes !="object"].index
# print(numeric_features) 测试输出
all_features[numeric_features]= all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)

all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape)
    # 完成编码 输出成功 12/09