import pickle
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import os.path as osp
import matplotlib.pyplot as plt
import xlrd
from torch.utils.data import DataLoader
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import sys
sys.path.append("..")
import func.process as pro
import func.cal as cal


time_start = time.time()                        # 程序开始时间
# 设定超参数
device = "cuda:1" if torch.cuda.is_available() else "cpu"
x_length = 1                   # 数据序列长度
y_length = 1                    # 标签序列长度，要求x_length是y_length的整数倍
lr = 0.0005                     # 学习率
weight_decay = 0.0005             # 权重衰减
epochs = 4000                    # 运行轮数
hidden_dim = 32                 # 隐藏层数量
gnn_style = "GraphSage"                   # RNN类型
save_fig = False                  # 是否保存图片
save_txt = False                 # 是否追加写入信息到txt
save_np = False
zh_en = False                   # 图片的标注、标题语言，True为中文，False为英文
ratio_train = 0.5               # 训练集所占比例
batch_size = 1

data_ad = '../dataset'

# su14-11-44A， su177
name = '苏14-11-44A'
df = pro.get_sheet(osp.join(data_ad, 'product_data'), name)             # 动态数据，包括日期、产量、套压、油压
dgo = df.loc[:, '日产气量(10⁴m³/d)'].values.reshape(-1)

g_address = osp.join("../graph/rnn_one_well", name)         # 保存图片的文件夹
re_address = osp.join("../result/rnn_one_well", name)

if not osp.exists(g_address):
    os.makedirs(g_address)
if not osp.exists(re_address):
    os.makedirs(re_address)

x = dgo
num = x.shape[0]
num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]     # get training dataset and test dataset

# Using Graph Neural network, prepare data information
x_train, y_train = cal.create_inout_sequences(data_train, x_length, y_length, style="arr")
x_test, y_test = cal.create_inout_sequences(data_test, x_length, y_length, style="arr")

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
num_nodes = x_train.shape[0] + x_test.shape[0]
num_train = x_train.shape[0]

x = torch.cat((x_train, x_test), dim=0)
y = torch.cat((y_train, y_test), dim=0)

adm = cal.path_graph(num_nodes)
# adm = cal.ts_un(num_nodes, 6)
edge_index, edge_weight = cal.tran_adm_to_edge_index(adm)

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = cal.index_to_mask(train_index, num_nodes).to(device)
test_mask = cal.index_to_mask(test_index, num_nodes).to(device)

# Using ResGraphNet, predicting time series (The Proposed Network Model)
model = cal.GNNTime(x_length, hidden_dim, y_length, edge_weight, gnn_style, num_nodes).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
edge_index = edge_index.to(device)

start_time = datetime.datetime.now()
print("Running, {}".format(gnn_style))
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    output_train, y_train = output[train_mask], y[train_mask]
    train_loss = criterion(output_train[:, -1], y_train[:, -1])
    train_loss.backward()
    optimizer.step()

    model.eval()
    output_test, y_test = output[test_mask], y[test_mask]
    test_loss = criterion(output_test[:, -1], y_test[:, -1])

    train_true = y_train.detach().cpu().numpy()[:, -1]
    train_pred = output_train.detach().cpu().numpy()[:, -1]
    test_true = y_test.detach().cpu().numpy()[:, -1]
    test_pred = output_test.detach().cpu().numpy()[:, -1]

    r2_train = cal.get_r2_score(train_pred, train_true, axis=1)
    r2_test = cal.get_r2_score(test_pred, test_true, axis=1)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))

# 只绘制预测的结果及其真实值
length_train, length_test = train_true.shape[0], test_true.shape[0]     # 训练集和测试集的长度
range_train = np.arange(length_train)                                   # 绘图时，训练集的横轴坐标
range_test = np.arange(length_train, length_train + length_test)        # 绘图时，测试集的横轴坐标

# 计算输出与输入的均方根误差
rmse_train_lstm = pro.get_rmse_two_array(a1=train_true, a2=train_pred)
rmse_test_lstm = pro.get_rmse_two_array(a1=test_true, a2=test_pred)

# 绘制原始输入的日产气量、网络输出日产气量，图像，并保存
plt.figure(figsize=(12, 8))
i = 0
plt.plot(range_train, train_true, label="训练集，原始数据", alpha=0.5)
plt.plot(range_train, train_pred, label="训练集，预测结果", alpha=0.5)
plt.plot(range_test, test_true, label="测试集，原始数据", alpha=0.5)
plt.plot(range_test, test_pred, label="测试集，预测结果", alpha=0.5)
# title = well_name + ", " + style + "\nTrain RMSE={:.5f}, Test RMSE={:.5f}".format(rmse_train, rmse_test)
# plt.title(title, fontsize=20)
plt.ylabel("日产气量("rf'$10^{{{4}}}m^{{{3}}}$'")", fontsize=20)
plt.xlabel("日期", fontsize=20)
plt.legend(fontsize=15)
image_address = osp.join(g_address, name + "_" + gnn_style + ".png")
if save_fig:
    plt.savefig(image_address)

print()
plt.show()
print()
