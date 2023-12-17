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

print(train_data.shape)
print(test_data.shape)
# 可以读取导数据 ok 12/09

#print(train_data.iloc[0:4,[0,1,2,3,-3,-22,-1]])
    #数据检查成功 ok 12/09


#2.训练集和测试集进行拼接
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)
# print(all_features.head())
    #测试数据拼接输出成功 ok 12/09

# 3.处理数据，对数据进行归一化 赋予缺失均值，离散变量转独热编码
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features) 测试输出
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
    # 完成编码 输出成功 12/09



# 4，数据转化为pytorch格式 编写其他方法
n_train = train_data.shape[0]

train_features = torch.tensor(
    all_features[:n_train].values.astype(np.float32),dtype=torch.float32
)
test_features=torch.tensor(
    all_features[:n_train].values.astype(np.float32),dtype=torch.float32
)
train_labels=torch.tensor(
    train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32
)
# print(train_data.SalePrice.values.reshape(-1,1).shape)

    # 出错 卡住 出现报错
    # TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    # 返回检查 12/09

    # 12/10 pandas 版本问题 或尝试修改代码
    # 修改代码添加 .astype(np.float32) 后正常


# 5.添加均方误差损失函数
loss = nn.MSELoss()
in_features = train_features.shape[1]
print(in_features)

# 定义网络，输出预测的一个房价
def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net


def log_rmse(net, features, labels):
    # 稳定数据，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()
# 代码先写这么多，补习一下数学基础 回顾内容
# 以及临时抱佛脚一下六级考试 暂停几天


#考完六级回来了 总计 考试体验非常好，明年还会继续考 继续



# 训练函数 12/17
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad() #优化器归零
            l = loss(net(X), y) #计算损失 网络 和标签
            l.backward() #反向传播
            optimizer.step() #梯度更新
        train_ls.append(log_rmse(net, train_features, train_labels)) #瞬时值添加到数组
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels)) #测试验证
    return train_ls, test_ls #返回损失值
