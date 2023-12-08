import GetHouseData
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

train_file = "kaggle_house_train"
test_file = "kaggle_house_test"

#读取数据
train_data = pd.read_csv(d2l.download(train_file))
test_data = pd.read_csv(d2l.download(test_file))
print(test_data.shape)
print(test_data.shape)

#训练集和测试集进行拼接