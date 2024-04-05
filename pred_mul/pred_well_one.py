"""
For one well, Production Prediction
"""
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
import sys
sys.path.append('..')
import func.process as pro
import func.net as net


GNN_STYLES = ["gcn"]
RNN_STYLES = ["lstm"]

time_start = time.time()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
l_x = 12
l_y = 1
lr = 0.0005
weight_decay = 0.0005
epochs = 1
hid_dim = 64
style = "gcn"
adm_style = "ts_un"
train_ratio = 0.5
batch_size = 32

well_name = '苏177'
data_dir = "dataset"
g_dir = osp.join("../graph/well_one", well_name)
re_dir = osp.join("../result/well_one", well_name)

"""
Data preparation, including Date, Production, Casing Pressure, Oil Pressure
"""
df = pro.get_sheet(osp.join(data_dir, 'product_data'), well_name)
data = df.loc[:, '日产气量(10⁴m³/d)'].values.reshape(-1)
train_loader, test_loader = pro.get_loader(data, l_x, l_y, train_ratio, batch_size)

"""
Model Training and Testing
"""
if style in GNN_STYLES:
    model = net.GNN(style, "ts_un", 1, num_nodes, device)
elif style in RNN_STYLES:
    model = net.GNN(style, "ts_un", 1, num_nodes, device)
else:
    raise TypeError("Unknown 'style', got {}".format(style))
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(epochs):
    train_pred, train_true, test_pred, test_true = [], [], [], []

    model.train()
    for item_train, (x_train, y_train, _) in enumerate(train_loader):

        x_train = x_train.to(device)           # 数据序列
        y_train = y_train.to(device)           # 标签序列

        optimizer.zero_grad()
        o_train = model(x_train)
        train_loss = criterion(o_train, y_train)                   # 一个序列的损失
        train_loss.backward()
        optimizer.step()

        train_pred_one = o_train.detach().cpu().numpy()
        train_true_one = y_train.detach().cpu().numpy()
        if item_train == 0:
            train_pred = train_pred_one
            train_true = train_true_one
        else:
            train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
            train_true = np.concatenate((train_true, train_true_one), axis=0)

    model.eval()
    for item_test, (x_test, y_test, _) in enumerate(test_loader):

        x_test = x_test.to(device)
        y_test = y_test.to(device)

        o_test = model(x_test)
        test_loss = criterion(o_test, y_test)

        test_pred_one = o_test.detach().cpu().numpy()
        test_true_one = y_test.detach().cpu().numpy()
        if item_test == 0:
            test_pred = test_pred_one
            test_true = test_true_one
        else:
            test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
            test_true = np.concatenate((test_true, test_true_one), axis=0)

    rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
    rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
    r2_train = net.cal_r2_one_arr(train_true, train_pred)
    r2_test = net.cal_r2_one_arr(test_true, test_pred)

    print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
          format(epoch, rmse_train, rmse_test, r2_train, r2_test))
