# 对单口油井，使用LSTM和机器学习，异常检测（简易版）
import pickle
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
import xlrd
from torch.utils.data import DataLoader
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import sys
sys.path.append("..")
import func.process as pro


def get_sheet(xls_ad, name):
    name_list = os.listdir(xls_ad)
    num_xls = len(name_list)
    df = None
    for i in range(num_xls):
        xls_name = name_list[i]
        xls = xlrd.open_workbook(osp.join(xls_ad, xls_name), on_demand=True)
        sheet_names = np.array(xls.sheet_names())
        if np.argwhere(sheet_names == name).reshape(-1).shape[0] != 0:
            df = pd.read_excel(osp.join(xls_ad, xls_name), sheet_name=name)
            break
    if df is None:
        raise ValueError('该油井不在已记录的动态数据中！')
    return df


time_start = time.time()                        # 程序开始时间
# 设定超参数
device = "cuda:1" if torch.cuda.is_available() else "cpu"
x_length = 1                   # 数据序列长度
y_length = 1                    # 标签序列长度，要求x_length是y_length的整数倍
lr = 0.0005                     # 学习率
weight_decay = 0.0005             # 权重衰减
epochs = 1                    # 运行轮数
hidden_dim = 64                 # 隐藏层数量
style = "LSTM"                   # RNN类型
at_style = "alpha"                 # 注意力机制
num_layers = 1                  # RNN层数
save_fig = False                  # 是否保存图片
save_txt = False                 # 是否追加写入信息到txt
save_np = False
zh_en = False                   # 图片的标注、标题语言，True为中文，False为英文
ratio_train = 0.5               # 训练集所占比例
threshold = 0.1              # 均方根误差阈值，大于时抛出warning
batch_size = 1

data_ad = '../dataset'

name = '苏177'
df = get_sheet(osp.join(data_ad, 'product_data'), name)             # 动态数据，包括日期、产量、套压、油压
dgo = df.loc[:, '日产气量(10⁴m³/d)'].values.reshape(-1)

g_address = osp.join("../graph/rnn_one_well", name)         # 保存图片的文件夹
re_address = osp.join("../result/rnn_one_well", name)

if not osp.exists(g_address):
    os.makedirs(g_address)
if not osp.exists(re_address):
    os.makedirs(re_address)

# 根据采集时间，将使用的数据集划分为训练集与测试集，并生成用于训练RNN的序列
data_used = dgo
# data_used = np.hstack([data, cp])
num_time = data_used.shape[0]                           # 使用的数据集中采集时间的数量
num_train = int(ratio_train * num_time)                 # 训练集的数量
data_train, data_test = data_used[:num_train], data_used[num_train:num_time]      # 划分为训练集与测试集
x_train, y_train = pro.get_xy(data_train, x_length, y_length, 'arr', 0)      # 训练集序列
x_test, y_test = pro.get_xy(data_test, x_length, y_length, 'arr', 0)        # 测试集序列

x_train, x_test = torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float()
y_train, y_test = torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()

train_dataset = pro.SelfData(x_train, y_train)
test_dataset = pro.SelfData(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
使用LSTM网络，异常检测
"""
model = pro.LSTMConcise(1, hidden_dim, y_length, x_length, num_layers).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 训练
train_predict_epoch, test_predict_epoch = [], []
train_loss_list, test_loss_list = [], []        # 绘制train_loss_all、test_loss_all根据epochs的变化
for epoch in range(epochs):
    model.train()
    train_loss_all, test_loss_all = 0, 0
    train_predict, train_true, test_predict, test_true, num_warn = None, None, None, None, 0
    for item_train, (x_train, y_train, _) in enumerate(train_loader):

        x_train = x_train.to(device)           # 数据序列
        y_train = y_train.to(device)           # 标签序列

        optimizer.zero_grad()
        output_train = model(x_train)
        train_loss = criterion(output_train, y_train)                   # 一个序列的损失
        train_loss_all = train_loss_all + train_loss.item()             # 所有序列的损失
        train_loss.backward()
        optimizer.step()

        train_predict_one = output_train.detach().cpu().numpy()
        train_true_one = y_train.detach().cpu().numpy()
        if item_train == 0:
            train_predict = train_predict_one
            train_true = train_true_one
        else:
            train_predict = np.vstack([train_predict, train_predict_one])
            train_true = np.vstack([train_true, train_true_one])

    for item_test, (x_test, y_test, _) in enumerate(test_loader):
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        output_test = model(x_test)
        test_loss = criterion(output_test, y_test)
        test_loss_all = test_loss_all + test_loss.item()

        test_predict_one = output_test.detach().cpu().numpy()
        test_true_one = y_test.detach().cpu().numpy()
        if item_test == 0:
            test_predict = test_predict_one
            test_true = test_true_one
        else:
            test_predict = np.vstack([test_predict, test_predict_one])
            test_true = np.vstack([test_true, test_true_one])

        # rmse_test_one = pro.get_rmse_two_array(a1=test_true_one, a2=test_predict_one)
        # if rmse_test_one > threshold:  # 添加告警机制，若均方根误差大于设定的阈值，打印该时刻，及其误差
        #     num_warn = num_warn + 1  # 告警天数的数量
        #     mess = "Root mean square error of {} is greater than {}, idx: {}, num_warn: {}". \
        #         format(new.time_all_list[num_train + x_length + idx], threshold, idx, num_warn)
        #     warnings.warn(mess)

    print("Epoch: {:03d}  Loss_Train: {:.9f}  Loss_Test: {:.9f}".format(
        epoch, train_loss_all, test_loss_all))

if save_np:
    np.save(osp.join(re_address, "train_true.npy"), train_true)
    np.save(osp.join(re_address, "train_predict.npy"), train_predict)
    np.save(osp.join(re_address, "test_true.npy"), test_true)
    np.save(osp.join(re_address, "test_predict.npy"), test_predict)
    np.save(osp.join(re_address, "train_predict_epoch.npy"), np.array(train_predict_epoch))
    np.save(osp.join(re_address, "test_predict_epoch.npy"), np.array(test_predict_epoch))

train_true, train_predict = train_true[:, -1], train_predict[:, -1]
test_true, test_predict = test_true[:, -1], test_predict[:, -1]

# 只绘制预测的结果及其真实值
length_train, length_test = train_true.shape[0], test_true.shape[0]     # 训练集和测试集的长度
range_train = np.arange(length_train)                                   # 绘图时，训练集的横轴坐标
range_test = np.arange(length_train, length_train + length_test)        # 绘图时，测试集的横轴坐标

# 计算输出与输入的均方根误差
rmse_train_lstm = pro.get_rmse_two_array(a1=train_true, a2=train_predict)
rmse_test_lstm = pro.get_rmse_two_array(a1=test_true, a2=test_predict)

# 绘制原始输入的日产气量、网络输出日产气量，图像，并保存
plt.figure(figsize=(12, 8))
i = 0
plt.plot(range_train, train_true, label="训练集，原始数据", alpha=0.5)
plt.plot(range_train, train_predict, label="训练集，预测结果", alpha=0.5)
plt.plot(range_test, test_true, label="测试集，原始数据", alpha=0.5)
plt.plot(range_test, test_predict, label="测试集，预测结果", alpha=0.5)
# title = well_name + ", " + style + "\nTrain RMSE={:.5f}, Test RMSE={:.5f}".format(rmse_train, rmse_test)
# plt.title(title, fontsize=20)
plt.ylabel("日产气量("rf'$10^{{{4}}}m^{{{3}}}$'")", fontsize=20)
plt.xlabel("日期", fontsize=20)
plt.legend(fontsize=15)
image_address = osp.join(g_address, name + "_" + style + ".png")
if save_fig:
    plt.savefig(image_address)

print()
plt.show()
print()
