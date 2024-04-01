import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric.nn as gnn
import xlrd
import pickle

from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
# from torch_cluster import knn
# from pygsp.graphs import Graph
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from torch_geometric.nn import knn_graph

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def get_sheet(files_path, well_name):
    """
    get information in one sheet

    :param files_path:
    :param well_name:
    :return:
    """
    files = os.listdir(files_path)
    for file in files:
        file_path = osp.join(files_path, file)
        xls = xlrd.open_workbook(file_path, on_demand=True)
        sheet_names = np.array(xls.sheet_names())
        if np.argwhere(sheet_names == well_name).reshape(-1).shape[0] != 0:
            df = pd.read_excel(file_path, sheet_name=well_name)
            return df
    raise ValueError('该油井不在已记录的动态数据中！')


def get_xy(data, l_x, l_y):
    """
    With sliding window, get x (input data) and y (input label)

    :param data:
    :param l_x: the length of x
    :param l_y: the length of y
    :return:
    """
    x_, y_ = None, None
    l = len(data)
    l_xy = l_x + l_y
    for i in range(l - l_xy + 1):
        if data.ndim == 1:
            x = data[i: (i + l_x)]
            y = data[(i + l_x): (i + l_x + l_y)]
        elif data.ndim == 2:
            x = data[i: (i + l_x), :]
            y = data[(i + l_x): (i + l_x + l_y), :]
        else:
            raise ValueError("!")
        x_, y_ = np.concatenate((x_, x), axis=0), np.concatenate((y_, y), axis=0)
    return x_, y_


def get_loader(data, l_x, l_y, train_ratio, batch_size):
    """
    Get training or testing dataloader

    :return:
    """
    num = data.shape[0]
    num_train = int(num * train_ratio)
    data_train, data_test = data[:num_train], data[num_train:]
    x_train, y_train = get_xy(data_train, l_x, l_y)
    x_test, y_test = get_xy(data_test, l_x, l_y)

    x_train, x_test = torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float()
    y_train, y_test = torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()
    train_dataset = SelfData(x_train, y_train)
    test_dataset = SelfData(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SelfData(Dataset):
    def __init__(self, data, label, *args):
        super(SelfData, self).__init__()
        self.data = be_tensor(data)
        self.label = be_tensor(label)
        self.args = args
        self.data_else = self.get_data_else()

    def get_data_else(self):
        num = len(self.args)
        data_else = [0] * num
        if num != 0:
            for i in range(num):
                data_else_one = self.args[i]
                data_else[i] = data_else_one
        return data_else

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        data_one = get_item_by_dim(self.data, item)
        label_one = get_item_by_dim(self.label, item)
        result = [data_one, label_one]
        if len(self.data_else) != 0:
            num = len(self.data_else)
            data_else_one = [0] * num
            for i in range(num):
                x = self.data_else[i]
                x_one = get_item_by_dim(x, item)
                data_else_one[i] = x_one
            result = result + data_else_one
        result.append(item)
        return tuple(result)


def get_item_by_dim(data, item):
    if torch.is_tensor(data):
        n_dim = data.dim()
    elif type(data) == np.ndarray:
        n_dim = data.ndim
    else:
        raise TypeError("The input must be torch.tensor or numpy.ndarray!")
    if n_dim == 1:
        return data[item]
    elif n_dim == 2:
        return data[item, :]
    elif n_dim == 3:
        return data[item, :, :]
    elif n_dim == 4:
        return data[item, :, :, :]
    else:
        raise ValueError("Unknown dim() of input!")


def be_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x)
    elif torch.is_tensor(x):
        return x
    else:
        raise TypeError("x must be tensor or ndarray, but gut {}".format(type(x)))


"""
===========================================
"""
class Product(object):
    def __init__(self, names, dates, pts, cps, ops, dgos):
        super(Product, self).__init__()
        self.names = names
        self.dates = dates
        self.pts = pts
        self.cps = cps
        self.ops = ops
        self.dgos = dgos


def tran_datetime(dt):
    m = dt.shape[0]
    dt_str = []
    for i in range(m):
        dt_one = dt[i]
        ts_one = pd.to_datetime(str(dt_one))
        dt_str_one = ts_one.strftime('%Y.%m.%d')
        dt_str.append(dt_str_one)
    dt_str = np.array(dt_str)
    return dt_str


# 计算两个1维array的均方根误差
def get_rmse_two_array(a1, a2):
    m_1, m_2 = a1.reshape(-1).shape[0], a2.reshape(-1).shape[0]
    if m_1 < m_2:  # 只取各自的前m项，m为两个array中最小长度
        m = m_1
    else:
        m = m_2
    a1_m, a2_m = a1[:m], a2[:m]
    result = np.sqrt(np.sum(np.square(a1_m - a2_m)) / m)
    return result


# 绘制一个一维numpy数组，简单一点
def plot_array(a, label=None, title=None, x_name=None, y_name=None, fig_size=(12, 8), font_size=20, font_size_le=15,
               style='plot', s=20, marker=False):
    plt.figure(figsize=fig_size)
    if style == 'scatter':
        plt.scatter(x=np.linspace(1, a.shape[0], a.shape[0]), y=a, label=label, s=s)
    else:
        if marker:
            plt.plot(a, label=label, marker="o", markersize=10)
        else:
            plt.plot(a, label=label)
    plt.xlabel(x_name, fontsize=font_size)
    plt.ylabel(y_name, fontsize=font_size)
    if label is not None:
        plt.legend(fontsize=font_size_le)
    plt.title(title, fontsize=font_size)
    return None


class ProductDynamics(object):
    def __init__(self, name, root="production_dynamics", x_feature=None, feature="日产气量(10⁴m³/d)", rz=False):
        super(ProductDynamics, self).__init__()
        self.name = name
        self.root = root
        self.rz = rz
        self.path = osp.join(self.root, self.name + ".xls")
        self.x_feature = x_feature  # 用于ml或者dl的数据，的特征（包括日产气量以外的特征）
        self.df = pd.read_excel(self.path)
        if self.rz:
            self.df = self.remove_zero(feature=feature)
        self.data = self.get_data(feature=feature)  # 该井的日产气量
        self.length = self.data.shape[0]

    def remove_zero(self, feature="日产气量(10⁴m³/d)"):  # 对某一个特征进行除零操作
        df_remove_zero = self.df[~self.df[feature].isin([0.])]
        return df_remove_zero

    def plot(self, feature="日产气量(10⁴m³/d)", rz=False, label=None, title=None):
        if rz:
            df_rz = self.remove_zero(feature=feature)
            value = df_rz.loc[:, feature].values.reshape(-1)
            plot_array(a=value, title=title, label=label)
        else:
            value = self.df.loc[:, feature].values.reshape(-1)
            plot_array(a=value, title=title, label=label)

    def get_data(self, feature="日产气量(10⁴m³/d)"):
        x = self.df.loc[:, feature].values.reshape(-1)
        return x

    def get_x_data(self):
        x_data = self.df.loc[:, self.x_feature].values
        return x_data

    def train_test_split(self, ratio=0.75):
        num_train = int(self.length * ratio)
        if self.x_feature is None:
            data_train = self.data[:num_train]
            data_test = self.data[num_train:]
        else:
            data = self.df.loc[:, self.x_feature].values
            data_train = data[:num_train]
            data_test = data[num_train:]
        return data_train, data_test


# 使用某个回归器，来撸一个数据
def eval_on_features(regress, x, y, title, ratio=0.75, fig_size=(12, 8), alpha=0.5):
    num = x.shape[0]  # 输入样本的数量
    num_train = int(ratio * num)  # 训练集数量
    if (x.ndim == 2) & (y.ndim == 1):  # 输入数据为二维，输入标签为一维
        x_train, x_test = x[:num_train, :], x[num_train:, :]
        y_train, y_test = y[:num_train], y[num_train:]
        regress.fit(x_train, y_train)
        y_train_pred = regress.predict(x_train)
        y_test_pred = regress.predict(x_test)
        train_score = regress.score(x_train, y_train)
        test_score = regress.score(x_test, y_test)

        plt.figure(figsize=fig_size)
        title = title + "\nTrain Score={:.5f}, Test Score={:.5f}".format(train_score, test_score)
        plt.title(title, fontsize=20)
        range_train = np.arange(num_train)  # 训练集绘图的横坐标取值
        range_test = np.arange(num_train, num)  # 测试集绘图的横坐标取值
        plt.plot(range_train, y_train, label="train", alpha=alpha)
        plt.plot(range_train, y_train_pred, label="train predict", alpha=alpha)
        plt.plot(range_test, y_test, label="test", alpha=alpha)
        plt.plot(range_test, y_test_pred, label="test predict", alpha=alpha)
        plt.legend(fontsize=15)
        plt.ylabel("Daily Production", fontsize=20)
        plt.xlabel("Date", fontsize=20)
    else:
        raise ValueError("x.shape and y.shape can't be calculated")
    return train_score, test_score


# 使用某个回归器，输入训练集数据、训练集标签、测试集数据，测试集标签
def eval_on_features_non_ratio(regress, x_train, y_train, x_test, y_test, title, score=False, mse=False, rmse=True,
                               fig_size=(12, 8), alpha=0.5, zh_en=True):
    train_return, test_return = 0, 0  # 训练集、测试集，评价性能指标
    if (x_train.ndim == 3) & (y_train.ndim == 2) & (y_train.shape[1] == 1):
        """
        由create_inout_sequences生成的数据集，用于和LSTM/GRU进行算法上的比对，此时y_length=1，即预测之后一天的日产气量
        """
        style = 1
        x_train = np.squeeze(x_train, axis=2)
        x_test = np.squeeze(x_test, axis=2)
        y_train = np.squeeze(y_train, axis=1)
        y_test = np.squeeze(y_test, axis=1)
        num_train, num_test = x_train.shape[0], x_test.shape[0]  # 训练集与测试集样本数量
        num = num_train + num_test
        regress.fit(x_train, y_train)
        y_train_pred = regress.predict(x_train)  # 对训练集输入数据的预测结果，即训练集输出
        y_test_pred = regress.predict(x_test)  # 对测试集输入数据的预测结果，即测试集输出

    elif (x_train.ndim == 3) & (y_train.ndim == 2) & (y_train.shape[1] != 1) & (x_train.shape[1] >= y_train.shape[1]):
        """
        由create_inout_sequences生成的数据集，用于和RNN进行算法上的比对，此时y_length!=1且x_length>=y_length，预测之后多天的日产气量
        因此，需要将输入数据展开成2维，输入标签展开成1维
        """
        style = 2
        num_train, num_test = x_train.shape[0], x_test.shape[0]  # 训练集与测试集样本数量
        num = num_train + num_test  # 所有样本的数量
        x_train, x_test = np.squeeze(x_train, axis=2), np.squeeze(x_test, axis=2)  # 将数据从3维压缩到2维
        x_length, y_length = x_train.shape[1], y_train.shape[1]  # 输入的，数据长度和标签长度
        y_train_shape, y_test_shape = y_train.shape, y_test.shape  # 保留原本标签的形状
        div, mod = int(x_length // y_length), int(x_length % y_length)  # 计算两个长度的商和余数
        if mod != 0:
            raise ValueError("If x_length >= y_length, x_length must be an integral multiple of y_length")
        y_train_row, y_test_row = y_train.reshape(-1), y_test.reshape(-1)
        x_train_row, x_test_row = x_train.reshape(-1, div), x_test.reshape(-1, div)
        regress.fit(x_train_row, y_train_row)
        y_train_row_pred = regress.predict(x_train_row)  # 对训练集输入数据的预测结果，即训练集输出
        y_test_row_pred = regress.predict(x_test_row)  # 对测试集输入数据的预测结果，即测试集输出

        y_train_pred = y_train_row_pred.reshape(y_train_shape)  # 将展开后计算的标签变为原来的形状
        y_test_pred = y_test_row_pred.reshape(y_test_shape)
    elif (x_train.ndim == 3) & (y_train.ndim == 2) & (x_train.shape[1] < y_train.shape[1]):
        """
        由create_inout_sequences生成的数据集，用于和RNN进行算法上的比对，此时x_length<y_length，预测之后多天的日产气量
        因此，需要多次训练回归器，分别预测结果后拼接
        """
        style = 3
        num_train, num_test = x_train.shape[0], x_test.shape[0]  # 训练集与测试集样本数量
        num = num_train + num_test  # 所有样本的数量
        x_train, x_test = np.squeeze(x_train, axis=2), np.squeeze(x_test, axis=2)  # 将数据从3维压缩到2维
        x_length, y_length = x_train.shape[1], y_train.shape[1]  # 输入的，数据长度和标签长度
        div, mod = int(y_length // x_length), int(y_length % x_length)  # 计算两个长度的商和余数
        if mod != 0:
            raise ValueError("If x_length < y_length, y_length must be an integral multiple of x_length")
        for i in range(div):
            y_train_one, y_test_one = y_train[:, i], y_test[:, i]
            regress.fit(x_train, y_train_one)
            y_train_one_pred = regress.predict(x_train).reshape(-1, 1)
            y_test_one_pred = regress.predict(x_test).reshape(-1, 1)
            if i != 0:
                y_train_pred = np.hstack([y_train_pred, y_train_one_pred])
                y_test_pred = np.hstack([y_test_pred, y_test_one_pred])
            else:
                y_train_pred = y_train_one_pred
                y_test_pred = y_test_one_pred
    else:
        raise TypeError("input dim is not corrected")

    if score:  # 标题添加决定系数score
        train_score = regress.score(x_train, y_train)  # 决定系数
        test_score = regress.score(x_test, y_test)
        title = title + "\nTrain Score={:.5f}, Test Score={:.5f}".format(train_score, test_score)
        train_return, test_return = train_score, test_score

    if mse:  # 计算并在标题添加均方损失（与LSTM比较）
        output_train_t, y_train_t = torch.from_numpy(y_train_pred), torch.from_numpy(y_train)
        output_test_t, y_test_t = torch.from_numpy(y_test_pred), torch.from_numpy(y_test)
        criterion = torch.nn.MSELoss()
        loss_train, loss_test = 0, 0
        for i in range(num_train):
            output_train_t_one, y_train_t_one = output_train_t[i], y_train_t[i]
            loss_train_one = criterion(output_train_t_one, y_train_t_one)
            loss_train = loss_train + loss_train_one.item()
        for i in range(num_test):
            output_test_t_one, y_test_t_one = output_test_t[i], y_test_t[i]
            loss_test_one = criterion(output_test_t_one, y_test_t_one)
            loss_test = loss_test + loss_test_one.item()
        title = title + "\nTrain Loss={:.5f}， Test Loss={:.5f}".format(loss_train, loss_test)
        train_return, test_return = loss_train, loss_test

    if rmse:  # 均方根误差
        rmse_train = get_rmse_two_array(a1=y_train, a2=y_train_pred)
        rmse_test = get_rmse_two_array(a1=y_test, a2=y_test_pred)
        train_return, test_return = rmse_train, rmse_test

    plt.figure(figsize=fig_size)  # 绘制原始数据、预测结果
    plt.title(title, fontsize=20)
    range_train = np.arange(num_train)  # 训练集绘图的横坐标取值
    range_test = np.arange(num_train, num)  # 测试集绘图的横坐标取值
    if zh_en:  # 中文
        if style == 1:
            plt.plot(range_train, y_train, label="训练集，原始数据", alpha=alpha)
            plt.plot(range_train, y_train_pred, label="训练集，预测结果", alpha=alpha)
            plt.plot(range_test, y_test, label="测试集，原始数据", alpha=alpha)
            plt.plot(range_test, y_test_pred, label="测试集，输出结果", alpha=alpha)
        elif (style == 2) | (style == 3):
            plt.plot(range_train, y_train[:, 0], label="训练集，原始数据", alpha=0.5)
            plt.plot(range_train, y_train_pred[:, 0], label="训练集，预测结果", alpha=0.5)
            plt.plot(range_test, y_test[:, 0], label="测试集，原始数据", alpha=0.5)
            plt.plot(range_test, y_test_pred[:, 0], label="测试集，预测结果", alpha=0.5)
    else:  # 英文
        if style == 1:
            plt.plot(range_train, y_train, label="Train, original data", alpha=alpha)
            plt.plot(range_train, y_train_pred, label="Train, predict result", alpha=alpha)
            plt.plot(range_test, y_test, label="Test, original data", alpha=alpha)
            plt.plot(range_test, y_test_pred, label="Test, predict result", alpha=alpha)
        elif (style == 2) | (style == 3):
            plt.plot(range_train, y_train[:, 0], label="Train, original data", alpha=0.5)
            plt.plot(range_train, y_train_pred[:, 0], label="Train, predict result", alpha=0.5)
            plt.plot(range_test, y_test[:, 0], label="Test, original data", alpha=0.5)
            plt.plot(range_test, y_test_pred[:, 0], label="Test, predict result", alpha=0.5)
    plt.legend(fontsize=15)
    plt.ylabel("Daily Gas Production("rf'$10^{{{4}}}m^{{{3}}}$'")", fontsize=20)
    plt.xlabel("Date", fontsize=20)
    return train_return, test_return


# 多个文件
class ProductDynamicsSome(object):
    def __init__(self, files, root="production_dynamics", rz=False, rz_all=False, ratio=0.75):
        super(ProductDynamicsSome, self).__init__()
        self.files = files  # 各个文件的名称，类型list
        self.num_files = len(files)  # 文件的数量
        self.root = root  # 文件所在目录
        self.rz = rz  # 文件各自除零后，在拼接
        self.rz_all = rz_all  # 某个特征的数据拼接后，再除零
        self.df = self.read_file()
        self.min_length = self.smallest_length()
        self.data = self.concat_data()
        self.ratio = ratio  # 训练集所占比例，测试集为剩余样本

    def read_file(self, feature="日产气量(10⁴m³/d)"):  # 读取各个文件，选择是否除零
        df = []
        for i in range(self.num_files):
            file_one = self.files[i] + ".xls"
            file_address = osp.join(self.root, file_one)
            df_one = pd.read_excel(file_address)
            if self.rz:
                df_one = df_one[~df_one[feature].isin([0.])]
            df.append(df_one)
        return df

    def concat_data(self, feature="日产气量(10⁴m³/d)"):  # 将各个文件的某个特征（默认为日产气量）拼接
        value = []
        for i in range(self.num_files):
            df_one = self.df[i]
            value_one = df_one.loc[:, [feature]].values
            value_one = value_one[:self.min_length]  # 选取文件中日产气量长度最小的，作为拼接后数据的长度
            if i != 0:
                value = np.hstack([value, value_one])
            else:
                value = value_one
        return value

    def smallest_length(self):  # 最少的样本总数
        length_min = float('inf')
        for i in range(self.num_files):
            length_one = self.df[i].shape[0]
            if length_one < length_min:
                length_min = length_one
        return length_min

    def train_test_split_torch(self):  # 划分训练集与测试集，用于pytorch
        num_train = int(self.min_length * self.ratio)
        data_train = self.data[:num_train, :]
        data_test = self.data[num_train:, :]
        return data_train, data_test

    def create_inout_sequences(self, style, x_length=12, y_length=1):  # 创建输入、输出数据集
        seq = []
        data_train, data_test = self.train_test_split_torch()
        if style == "train":
            data = data_train
        elif style == "test":
            data = data_test
        else:
            raise ValueError('style must be "train" or "test", but got {}'.format(style))
        data_length = len(data)
        x_y_length = x_length + y_length
        for i in range(data_length - x_y_length + 1):
            seq_one = data[i:i + x_length, :]
            label_one = data[i + x_length:i + x_length + y_length, :]
            seq.append((seq_one, label_one))
        return seq

    def plot_all(self, feature="日产气量(10⁴m³/d)"):  # 绘制所有文件的那个数据
        for i in range(self.num_files):
            df_one = self.df[i]
            data_one = df_one.loc[:, [feature]].values
            plot_array(a=data_one, title=self.files[i])


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop, style):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop  # 是否增加丢弃层
        self.style = style  # RNN的类型
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.gru1 = nn.GRU(input_dim, hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        if self.style == "LSTM":
            out_put, (h, c) = self.lstm1(x)
            out_put, (h, c) = self.lstm2(out_put, (h, c))
        elif self.style == "GRU":
            out_put, h = self.gru1(x)
            out_put, h = self.gru2(out_put, h)
        else:
            raise TypeError("style must be 'LSTM' or 'GRU', but got {}".format(self.style))
        out_put = out_put.view(len(x), -1)
        out_put = F.relu(out_put)
        if self.drop:
            out_put = F.dropout(out_put)

        out_put = self.linear(out_put)
        out_put = torch.mean(out_put, dim=0)
        return out_put


# 用于rnn_one_concise.py的LSTM简易版
class LSTMConcise(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, x_length, num_layers=1, style="LSTM", at_style=None,
                 u_length=5):
        super(LSTMConcise, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.x_length = x_length
        self.num_layers = num_layers
        self.style = style  # RNN类型
        self.at_style = at_style  # 注意力机制
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.other = nn.Sequential(nn.ReLU(), nn.Dropout())  # 激活函数+丢弃层
        self.tran_nn = nn.Linear(x_length, output_dim)
        self.attn_weight = nn.Parameter(torch.Tensor(num_layers, 1))

        self.u_length = u_length
        self.w_alpha = nn.Parameter(torch.Tensor(num_layers, hidden_dim, u_length))  # 注意力机制的权重
        self.u = nn.Parameter(torch.Tensor(u_length, 1))
        self.b_alpha = nn.Parameter(torch.Tensor(num_layers, x_length, u_length))
        init.kaiming_uniform_(self.attn_weight)  # 初始化自定义的权重
        init.kaiming_uniform_(self.w_alpha)
        init.kaiming_uniform_(self.u)
        init.kaiming_uniform_(self.b_alpha)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.style == "LSTM":
            output, (h0, c0) = self.lstm(x)
            if self.at_style == "w_h":  # 添加注意力机制，暂时不会写
                h = h0.transpose(1, 2)
                attn_weight = self.attn_weight
                attn_weight = attn_weight.view(len(attn_weight), 1, 1)
                w_h = torch.bmm(h, attn_weight)
                w_h = w_h.transpose(1, 2)
                output = w_h
            elif self.at_style == "alpha":
                a = torch.bmm(h0, self.w_alpha)
                b = a + self.b_alpha
                b = torch.mean(b, dim=1)
                c = torch.tanh(b)
                d = torch.mm(c, self.u)
                e = torch.exp(d)
                e_sum = torch.sum(e)
                alpha = e / e_sum.item()

                h = h0.transpose(1, 2)
                alpha = alpha.unsqueeze(2)
                f = torch.bmm(h, alpha)
                w_alpha = f.transpose(1, 2)
                output = w_alpha

        elif self.style == "GRU":
            output, h0 = self.gru(x)
            if self.at_style is not None:
                pass
        else:
            raise TypeError("style must be 'LSTM' or 'GRU', but got {}".format(self.style))
        output = F.relu(output)
        output = output.squeeze(1)
        output = self.linear(output)
        # output = self.tran_xd_to_yd(output)
        return output

    def tran_xd_to_yd(self, x):  # 将数据从x的长度，转换成y的长度
        x_length = self.x_length
        y_length = self.output_dim
        num = self.input_dim  # 井的数量
        if x_length >= y_length:  # 如果x的长度大于等于y的长度，对x的长度按前后顺序求平均
            k = int(x_length / y_length)  # x长度与y长度的比例
            value_all = torch.zeros(size=(y_length, num)).to(device)
            for i in range(y_length):
                one = x[i: (i + k), :]
                value = torch.mean(one, dim=0)
                if num == 1:  # 单口油井
                    value_all[i] = value
                else:  # 多口油井
                    value_all[i, :] = value
            return value_all
        else:  # 若x的长度小于y的长度，构建一个全连接层，变换x的维度
            value_all = self.tran_nn(x)
            return value_all.transpose(0, 1)


# 训练多口井的日产气量
class LSTMSome(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop, n, y_length, style):
        super(LSTMSome, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop  # 是否增加丢弃层
        self.n = n  # 文件的数量
        self.y_length = y_length
        self.style = style  # 循环神经网络的类型
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim)
        self.gru1 = nn.GRU(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.n * self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        if self.style == "LSTM":
            out_put, (h, c) = self.lstm1(x)
            # out_put, (h, c) = self.lstm2(out_put, (h, c))
        elif self.style == "GRU":
            out_put, h = self.gru1(x)
        else:
            raise TypeError("style must be 'LSTM' or 'GRU', but got {}".format(self.style))

        out_put = out_put.view(len(x), -1)
        out_put = F.relu(out_put)
        if self.drop:
            out_put = F.dropout(out_put)

        out_put = self.linear(out_put)
        out_put = out_put[:self.y_length, :]
        return out_put


# 对基于NewWellInfo生成的数据集，构建循环神经网络（仅考虑日产气量）
class LSTMNew(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, x_length, style, ld1, rnn_2, num_layers=1):
        super(LSTMNew, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.x_length = x_length
        self.style = style  # RNN类型
        self.ld1 = ld1  # 标签是否仅考虑日产气量
        self.rnn_2 = rnn_2  # 是否施加两层RNN网络
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.other = nn.Sequential(nn.ReLU(), nn.Dropout())  # 激活函数+丢弃层

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.style == "LSTM":
            output, (h0, c0) = self.lstm(x)
            if self.rnn_2:
                output, (h1, c1) = self.lstm1(output, (h0, c0))
        elif self.style == "GRU":
            output, h0 = self.gru(x)
            if self.rnn_2:
                output, h1 = self.gru1(output, h0)
        else:
            raise TypeError("style must be 'LSTM' or 'GRU', but got {}".format(self.style))
        output = F.relu(output)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.tran_xd_to_yd(output)
        if self.ld1:
            output = torch.mean(output, dim=2)
        return output

    def tran_xd_to_yd(self, x):  # 将数据从x的长度各自平均，转换成y的长度
        x_length = self.x_length
        y_length = self.output_dim
        num = self.input_dim  # 井的数量
        k = int(x_length / y_length)  # x长度与y长度的比例
        value_all = torch.zeros(size=(y_length, num)).to(device)
        for i in range(y_length):
            one = x[i: (i + k), :]
            value = torch.mean(one, dim=0)
            if num == 1:  # 单口油井
                value_all[i] = value
            else:  # 多口油井
                value_all[i, :] = value
        return value_all


# 对基于NewWellInfo生成的数据集，构建循环神经网络并加入注意力机制（仅考虑日产气量）
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, x_length, style, ld1, rnn_2, num_seq, at_style, num_layers=1):
        super(LSTMAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.x_length = x_length
        self.style = style  # RNN类型
        self.ld1 = ld1  # 标签是否仅考虑日产气量
        self.rnn_2 = rnn_2  # 是否施加两层RNN网络
        self.num_seq = num_seq  # 序列的数量
        self.at_style = at_style  # 注意力机制的类型
        self.at_weight = None  # 用于注意力机制的权重
        self.h_weight = None  # 隐藏单元h的注意力权重
        self.c_weight = None  # 记忆单元c的注意力权重
        self.num_layers = num_layers  # RNN的层数
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.other = nn.Sequential(nn.ReLU(), nn.Dropout())  # 激活函数+丢弃层
        self.get_at_weight()  # 按照不同类型，初始化注意力权重

    def forward(self, x, idx):
        x = x.unsqueeze(1)
        if self.style == "LSTM":  # rnn类型为LSTM
            if self.rnn_2:  # 两层rnn
                if self.at_style == "1d":  # 第一种注意力机制
                    x_new = self.at_weight[idx] * x
                    output, (h, c) = self.lstm(x_new)
                    output, (_, _) = self.lstm1(output, (h, c))
                elif self.at_style == "2d":  # 第二种注意力机制
                    x_new = torch.zeros(size=x.shape).to(device)
                    for i in range(self.x_length):
                        x_new[i, :, :] = x[i, :, :] * self.at_weight[i]
                    output, (h, c) = self.lstm(x_new)
                    output, (_, _) = self.lstm1(output, (h, c))
                elif self.at_style == "3d":  # 第三种注意力机制
                    output, (h, c) = self.lstm(x)
                    # h_new = torch.bmm(h, self.h_weight)
                    # c_new = torch.bmm(c, self.c_weight)
                    h_new, c_new = torch.zeros(size=h.shape).to(device), torch.zeros(size=c.shape).to(device)
                    for i in range(self.num_layers):
                        h_new[i, :, :] = h[i, :, :] * self.h_weight[i]
                        c_new[i, :, :] = c[i, :, :] * self.c_weight[i]
                    output, (_, _) = self.lstm1(output, (h_new, c_new))
                else:  # 不添加注意力机制
                    output, (h, c) = self.lstm(x)
                    output, (_, _) = self.lstm1(output, (h, c))
            else:  # 单层rnn
                if self.at_style == "1d":  # 此时没有第一种注意力机制
                    x_new = self.at_weight[idx] * x
                    output, (h, c) = self.lstm(x_new)
                elif self.at_style == "2d":  # 第二种注意力机制
                    x_new = torch.zeros(size=x.shape).to(device)
                    for i in range(self.x_length):
                        x_new[i, :, :] = x[i, :, :] * self.at_weight[i]
                    output, (h, c) = self.lstm(x_new)
                elif self.at_style == "3d":
                    raise TypeError("If rnn_2 is False, then at_style can't be 3d")
                else:  # 不添加注意力机制
                    output, (h, c) = self.lstm(x)

        elif self.style == "GRU":  # rnn类型为GRU
            if self.rnn_2:  # 两层rnn
                if self.at_style == "1d":  # 第一种注意力机制
                    x_new = self.at_weight[idx] * x
                    output, h = self.gru(x_new)
                    output, _ = self.gru1(output, h)
                elif self.at_style == "2d":  # 第二种注意力机制
                    x_new = torch.zeros(size=x.shape).to(device)
                    for i in range(self.x_length):
                        x_new[i, :, :] = x[i, :, :] * self.at_weight[i]
                    output, h = self.gru(x_new)
                    output, _ = self.gru1(output, h)
                elif self.at_style == "3d":  # 第三种注意力机制
                    output, h, c = self.gru(x)
                    h_new = torch.zeros(size=h.shape).to(device)
                    for i in range(self.num_layers):
                        h_new[i, :, :] = h[i, :, :] * self.h_weight[i]
                    output, _ = self.gru1(output, h_new)
                else:  # 不添加注意力机制
                    output, h = self.gru(x)
                    output, _ = self.gru1(output, h)
            else:  # 单层rnn
                if self.at_style == "1d":  # 此时没有第一种注意力机制
                    x_new = self.at_weight[idx] * x
                    output, h = self.gru(x_new)
                elif self.at_style == "2d":  # 第二种注意力机制
                    for i in range(self.x_length):
                        x[i, :, :] = x[i, :, :] * self.at_weight[i]
                    output, h = self.gru(x)
                elif self.at_style == "3d":
                    raise TypeError("If rnn_2 is False, then at_style can't be 3d")
                else:  # 步天价注意力机制
                    output, h = self.gru(x)
        else:
            raise TypeError("style must be 'LSTM' or 'GRU', but got {}".format(self.style))
        output = F.relu(output)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.tran_xd_to_yd(output)
        if self.ld1:
            output = torch.mean(output, dim=2)
        return output

    def tran_xd_to_yd(self, x):  # 将数据从x的长度各自平均，转换成y的长度
        x_length = self.x_length
        y_length = self.output_dim
        num = self.input_dim  # 井的数量
        k = int(x_length / y_length)  # x长度与y长度的比例
        value_all = torch.zeros(size=(y_length, num)).to(device)
        for i in range(y_length):
            one = x[i: (i + k), :]
            value = torch.mean(one, dim=0)
            if num == 1:  # 单口油井
                value_all[i] = value
            else:  # 多口油井
                value_all[i, :] = value
        return value_all

    def get_at_weight(self):  # 在不同注意力机制的类型下，初始化注意力的权重
        if self.at_style == "1d":  # 对不同idx的多个样本，给每一个样本施加一个权重
            self.at_weight = nn.Parameter(torch.ones(self.num_seq))
        elif self.at_style == "2d":  # 对一个样本，在数据序列x_length上，一个时刻施加一个权重
            self.at_weight = nn.Parameter(torch.ones(self.x_length))
        elif self.at_style == "3d":  # 多层num_layers时，对每一层的h以及c施加一个权重
            # self.h_weight = nn.Parameter(torch.rand(self.num_layers, self.hidden_dim, self.hidden_dim))
            # self.c_weight = nn.Parameter(torch.rand(self.num_layers, self.hidden_dim, self.hidden_dim))
            self.h_weight = nn.Parameter(torch.from_numpy(np.ones(self.num_layers).astype(np.float32)))
            self.c_weight = nn.Parameter(torch.from_numpy(np.ones(self.num_layers).astype(np.float32)))
        return None


# 对基于NewWellInfo生成的数据集，构建循环神经网络（考虑日产气量、套压、油压）
class LSTMNewF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, x_length, style, ld1, rnn_2):
        super(LSTMNewF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.x_length = x_length
        self.style = style
        self.ld1 = ld1
        self.rnn_2 = rnn_2  # 是否施加两层RNN网络
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.other = nn.Sequential(nn.ReLU(), nn.Dropout())  # 激活函数+丢弃层

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.style == "LSTM":
            output, (h0, c0) = self.lstm(x)
            if self.rnn_2:
                output, (h1, c1) = self.lstm1(output, (h0, c0))
        elif self.style == "GRU":
            output, h0 = self.gru(x)
            if self.rnn_2:
                output, h1 = self.gru1(output, h0)
        else:
            raise TypeError("style must be 'LSTM' or 'GRU', but got {}".format(self.style))
        output = F.relu(output)
        output = self.linear(output)
        output = self.tran_xd_to_yd(output)
        if self.ld1:
            output = torch.mean(output, dim=2)
        return output

    def tran_xd_to_yd(self, x):  # 将数据从x的长度各自平均，转换成y的长度
        x_length = self.x_length
        y_length = self.output_dim
        num = self.input_dim  # 井的数量
        k = int(x_length / y_length)  # x长度与y长度的比例
        value_all = torch.zeros(size=(y_length, x.size(1), num)).to(device)
        for i in range(y_length):
            one = x[i: (i + k), :]
            value = torch.mean(one, dim=0)
            if num == 1:  # 单口油井
                value_all[i] = value
            else:  # 多口油井
                value_all[i, :] = value
        return value_all


# 将rnn_some.py的井名列表well_list拼接为，文件夹的名称，即"name1_name2_name3..."
def tran_well_list_to_folder(well_list, ld1, all_well=False):
    if all_well:  # 若考虑所有的油井，名称为"all"，不然文件夹名称也太长惹
        folder_name = "all"
        if ld1:
            folder_name = folder_name + "_ld1"
    else:
        num_well = len(well_list)  # 油井的数量
        folder_name = ""
        for i in range(num_well):
            well_one = well_list[i]
            if i != (num_well - 1):  # 不是最后一口井
                folder_name = folder_name + well_one + "_"
            else:  # 最后一口井，不加"_"
                folder_name = folder_name + well_one
        if ld1:
            folder_name = folder_name + "_ld1"
    return folder_name


# rnn_some.py，生成总文件夹名称，比tran_well_list_to_folder函数生成的文件夹大一个级别
def get_rnn_folder(features, num):
    if num != 1:  # 预测多口井
        folder_name = "rnn_some_wells"
    else:  # 预测单口井
        folder_name = "rnn_one_well"
    if features == ["日产气量(10⁴m³/d)"]:  # 仅考虑日产气量
        pass
    elif features == ["日产气量(10⁴m³/d)", "套压(MPa)"]:
        folder_name = folder_name + "_with_cp"
    elif features == ["日产气量(10⁴m³/d)", "油压(MPa)"]:
        folder_name = folder_name + "_with_op"
    elif features == ["日产气量(10⁴m³/d)", "套压(MPa)", "油压(MPa)"]:
        folder_name = folder_name + "_with_cp_op"
    else:
        raise TypeError("Unknown type of features, got {}".format(features))
    return folder_name


def get_rnn_model(all_well, features, x_length, y_length, hidden_dim, num, num_nodes, style, ld1, rnn_2, num_layers):
    """
    rnn_some.py，根据不同情况，选择RNN神经网络模型，并实例化
    :param all_well: 是否考虑全部油井，类型bool
    :param features: 是否考虑其余动态特征，类型bool，str
    :param x_length: 数据序列长度，类型int
    :param y_length: 标签序列长度，类型int
    :param hidden_dim: 隐藏层数量，类型int
    :param num: 考虑的油井数量，类型int
    :param num_nodes: 所有油井数量，类型int
    :param style: RNN类型，类型'LSTM' or 'GRU'
    :param ld1: 标签维度是否为1，类型bool
    :param rnn_2: 是否施加两层RNN网络，类型bool
    :param num_layers: RNN层数，类型int
    :return: RNN模型
    """
    if all_well:  # 若考虑所有油井
        if features == ["日产气量(10⁴m³/d)"]:  # 仅考虑日产气量
            model = LSTMNew(num_nodes, hidden_dim, y_length, x_length, style, ld1, rnn_2).to(device)
        else:
            model = LSTMNewF(num_nodes, hidden_dim, y_length, x_length, style, ld1, rnn_2).to(device)
    else:  # 油井由well_list给定
        if features == ["日产气量(10⁴m³/d)"]:
            model = LSTMNew(num, hidden_dim, y_length, x_length, style, ld1, rnn_2).to(device)
        else:
            num_f = len(features)  # 动态特征的数量
            model = LSTMNewF(num_f, hidden_dim, y_length, x_length, style, ld1, rnn_2).to(device)
    return model


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop  # 是否添加丢弃层
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.linear1(x)
        h = F.relu(h)
        if self.drop:
            h = F.dropout(h)
        output = self.linear2(h)
        return output


# 读取一个文件所有的sheet
class ProductDynamicAllSheets(object):
    def __init__(self, file, root="production_dynamics", rz=False, rz_all=False, ratio=0.75):
        self.file = file  # 文件名称
        self.root = root  # 根目录
        self.file_address = osp.join(self.root, self.file)  # 文件存放位置
        self.df, self.length, self.sheet_names = self.read_sheets()  # 表格的数据、对应的名称
        self.num_sheets = len(self.sheet_names)

    def read_sheets(self):
        df, length = [], []
        wb = xlrd.open_workbook(self.file_address)
        sheet_names = wb.sheet_names()  # sheet名称
        for i in range(len(sheet_names)):  # 读取每个sheet
            df_one = pd.read_excel(self.file_address, sheet_name=sheet_names[i])
            length_one = df_one.shape[0]
            df.append(df_one)
            length.append(length_one)
        return df, length, sheet_names


# def get_xy(data, l_x, l_y, style, label_dim):
#     xy_, x_, y_ = [], None, None
#     l = len(data)
#     l_xy = l_x + l_y
#     if style == "list":     # list类型，用于RNN网络
#         for i in range(l - l_xy + 1):
#             if data.ndim == 2:
#                 x = data[i: (i + l_x), :]
#                 if label_dim is not None:
#                     y = data[(i + l_x): (i + l_x + l_y), label_dim].reshape(-1, 1)
#                 else:
#                     y = data[(i + l_x): (i + l_x + l_y), :]
#             elif data.ndim == 1:
#                 x = data[i: (i + l_x)]
#                 y = data[(i + l_x): (i + l_x + l_y)]
#             elif data.ndim == 3:
#                 x = data[i: (i + l_x), :, :]
#                 y = data[(i + l_x): (i + l_x + l_y), :, :]
#             else:
#                 raise TypeError('!')
#             xy_.append((x, y))
#         return xy_
#
#     elif style == "arr":    # array类型，用于机器学习
#         for i in range(l - l_xy + 1):
#             if data.ndim == 2:
#                 x = data[i: (i + l_x), :]
#                 if label_dim is not None:
#                     y = data[(i + l_x): (i + l_x + l_y), label_dim].reshape(-1, 1)
#                 else:
#                     y = data[(i + l_x): (i + l_x + l_y), :]
#             elif data.ndim == 1:
#                 x = data[i: (i + l_x)]
#                 y = data[(i + l_x): (i + l_x + l_y)]
#             else:
#                 raise TypeError('!')
#             x, y = x.reshape(1, -1), y.reshape(1, -1)
#             if i == 0:
#                 x_, y_ = x, y
#             else:
#                 x_, y_ = np.vstack([x_, data]), np.vstack([y_, label])
#         return x_, y_


# 获得一个xls所有sheet的名称
def get_file_sheets_name(file, root="production_dynamics"):
    file_address = osp.join(root, file + ".xls")  # 文件存放位置
    file_data = xlrd.open_workbook(file_address)
    sheet_names = file_data.sheet_names()  # sheet名称
    if sheet_names[-1] == "井名列表":
        sheet_names = sheet_names[:-1]  # 最后一个"井名列"暂时不需要
    return sheet_names


# 多口油井的信息
class WellInfo(object):
    def __init__(self, name, root="../dataset/static_data/new_SD", scale=None, gtype="knn", k=2, s=20, data=None,
                 time=None,
                 cp=None, op=None, drop_duplicates=True, fish=None):
        self.name = name  # 读取的文件名，不需要.xls
        self.root = root  # 文件所在目录
        address = osp.join(self.root, self.name + ".csv")  # 文件存放地址
        self.info = pd.read_csv(address)  # 全部信息
        self.drop_duplicates = drop_duplicates  # 是否去重（相同的井名）
        if self.drop_duplicates:
            self.info.drop_duplicates(subset=["井名"], keep="first")
        self.fish = fish
        if self.fish is not None:
            self.info = self.delete_well(self.fish)  # 删除漏网之鱼，看read_data()里的fish
        self.gtype = gtype  # 图的类型
        self.k = k  # knn图的近邻点数量
        self.xy = self.get_xy()  # 节点坐标
        self.scale = scale  # 缩放坐标的比例
        if self.scale is not None:
            self.xy = self.zoom(scale=self.scale)  # 缩放坐标
        self.x_range, self.y_range = self.xy_range()  # 坐标的范围
        self.s = s  # 图节点大小
        self.graph = self.get_graph()  # 生成图结构
        self.well_names = self.info.loc[:, "井名"].values.tolist()  # 井名

        # 判断data(日产气量),time(采集时间),cp(套压),op(油压)之前是否保存，若无，计算data、time、cp、op并保存为pkl格式
        if (data is None) | (time is None) | (cp is None) | (op is None):
            self.data, self.time, self.cp, self.op = self.read_data()
        else:  # 若已保存，按照pickle读取
            self.data, self.time, self.cp, self.op = self.pickle_load(data, time, cp, op)
        self.time_all_list, self.time_all_arr = get_all_time(self.time)  # 所有存在的时间
        self.data_all, self.cp_all, self.op_all = self.fill_time()  # 在所有非采集时间上，采集数据补充为零
        self.num_time = len(self.time_all_list)  # 采集时间的数量
        self.num_well = self.info.shape[0]  # 井的数量

    def get_xy(self):  # 获得坐标
        x = self.info.loc[:, ["井口纵坐标X"]].values
        y = self.info.loc[:, ["井口横坐标Y"]].values
        xy = np.hstack([x, y])
        return xy

    def xy_range(self):  # 计算坐标的范围
        x, y = self.xy[:, 0], self.xy[:, 1]
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        x_range, y_range = [x_min, x_max], [y_min, y_max]
        return x_range, y_range

    def zoom(self, scale):  # 缩放坐标
        scale_x, scale_y = scale[0], scale[1]
        x, y = self.xy[:, 0] / scale_x, self.xy[:, 1] / scale_y
        xy = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
        return xy

    def get_graph(self, labels=None):  # 创建图结构
        if self.gtype == "knn":  # k近邻图
            graph = KnnGraph(xy=self.xy, k=self.k, s=self.s, labels=labels)
        else:
            raise TypeError("gtype must be 'knn', but got {}".format(self.gtype))
        return graph

    def read_data(self):
        if self.name == "su47":
            file = ["47(1)", "47(2)", "47(3)", "47(4)", "47(5)", "47(6)", "47(7)", "47(8)", "47(9)", "47(10)", "47(11)"]
        elif self.name == "su120":
            file = ["120(1)", "120(2)"]
        elif self.name == "su14":
            file = ["苏14(1-100)", "苏14(101-200)", "苏14(201-300)", "苏14(301-400)", "苏14(401-500)", "苏14(501-600)",
                    "苏14(601-700)", "苏14(701-800)", "苏14(801-900)", "苏14(901-1000)", "苏14(1101-1200)",
                    "苏14(1201-1300)", "苏14(1301-1400)"]
        elif self.name == "su48":
            file = ["48(1)", "48(2)", "48(3)", "48(4)", "48(5)", "48(6)", "48(7)", "48(8)", "48(9)"]
        elif self.name == "tao2":
            file = ["桃2(1-200)", "桃2(201-400)", "桃2(401-600)", "桃2(601-800)", "桃2(601-984)"]
        elif self.name == "all":
            file = ["47(1)", "47(2)", "47(3)", "47(4)", "47(5)", "47(6)", "47(7)", "47(8)", "47(9)", "47(10)", "47(11)",
                    "120(1)", "120(2)", "苏14(1-100)", "苏14(101-200)", "苏14(201-300)", "苏14(301-400)",
                    "苏14(401-500)",
                    "苏14(501-600)", "苏14(601-700)", "苏14(701-800)", "苏14(801-900)", "苏14(901-1000)",
                    "苏14(1101-1200)", "苏14(1201-1300)", "苏14(1301-1400)", "48(1)", "48(2)", "48(3)", "48(4)",
                    "48(5)",
                    "48(6)", "48(7)", "48(8)", "48(9)", "桃2(1-200)", "桃2(201-400)", "桃2(401-600)", "桃2(601-800)",
                    "桃2(601-984)"]
        else:
            raise ValueError("name must be 'su14' or 'su47' or 'su48' or 'su120' or 'tao2' or 'all', but got {}".
                             format(self.name))
        num_files = len(file)  # 文件的数量
        sheet, sheet_group, length_group = [], [], []
        for i in range(num_files):
            file_one = file[i]
            sheet_one = get_file_sheets_name(file=file_one)
            length_sheet_one = len(sheet_one)
            length_group.append(length_sheet_one)  # 每个文件包含的sheet数量
            sheet = sheet + sheet_one  # 不分组
            sheet_group.append(sheet_one)  # 分组

        fish = list(set(self.well_names) - set(sheet))  # 检查漏网之鱼，并保存为txt
        fish_arr = np.array(fish)
        fish_save_address = osp.join(self.root, self.name + "_fish.txt")
        if self.fish is None:
            np.savetxt(fish_save_address, fish_arr, fmt="%s")
        """
        在su47里但不在47(?)里：['苏47-7-85']
        在su14里但不在苏14(?)里：su14_fish.txt
        """

        sheet = np.array(sheet)  # 获得所有井名的列表
        num_wells = sheet.shape[0]  # 井的数量
        data, time, cp, op = [], [], [], []  # 初始化
        for i in range(num_wells):
            well = self.well_names[i]
            index = np.argwhere(sheet == well)
            row, col = index_in_sheet_group(sheet_group, length_group, index)
            file_to_well = file[row]  # 对应的文件名
            sheet_one = sheet_group[row][col]  # 对应的sheet
            file_address = osp.join("production_dynamics", file_to_well + ".xls")
            df_one = pd.read_excel(file_address, sheet_name=sheet_one)
            data_one = df_one.loc[:, "日产气量(10⁴m³/d)"].values.reshape(-1)  # 数据
            data.append(data_one)
            time_one = df_one.loc[:, "日期"].values.reshape(-1)  # 时刻
            time.append(time_one)
            cp_one = df_one.loc[:, "套压(MPa)"].values.reshape(-1)  # 套压
            cp.append(cp_one)
            op_one = df_one.loc[:, "油压(MPa)"].values.reshape(-1)  # 油压
            op.append(op_one)

        # 保存data、time、cp、op
        data_save_name = osp.join(self.root, self.name + "data" + ".pkl")
        with open(data_save_name, "wb") as f_data:
            pickle.dump(data, f_data)
        time_save_name = osp.join(self.root, self.name + "time" + ".pkl")
        with open(time_save_name, "wb") as f_time:
            pickle.dump(time, f_time)
        cp_save_name = osp.join(self.root, self.name + "cp" + ".pkl")
        with open(cp_save_name, "wb") as f_cp:
            pickle.dump(cp, f_cp)
        op_save_name = osp.join(self.root, self.name + "op" + ".pkl")
        with open(op_save_name, "wb") as f_op:
            pickle.dump(op, f_op)
        return data, time, cp, op

    def delete_well(self, name):  # 删去漏网之鱼（不知道为啥，若fish只有一个，无法读取?_fish.txt，参考su48）
        info = self.info
        fish_address = osp.join(self.root, name + ".txt")
        fish = np.loadtxt(fish_address, dtype=np.str, encoding="GBK")
        if fish.size == 0:  # 读取了空文件
            pass
        else:
            fish = fish.reshape(-1)
            for i in range(fish.shape[0]):
                name_one = fish[i]
                info = info[~info["井名"].isin([name_one])]
        return info

    def pickle_load(self, data, time, cp, op):  # 读取data、time、cp、op，pkl格式
        data_address = osp.join(self.root, data + ".pkl")
        time_address = osp.join(self.root, time + ".pkl")
        cp_address = osp.join(self.root, cp + ".pkl")
        op_address = osp.join(self.root, op + ".pkl")
        with open(data_address, "rb") as f_data:
            data = pickle.load(f_data)
        with open(time_address, "rb") as f_time:
            time = pickle.load(f_time)
        with open(cp_address, "rb") as f_cp:
            cp = pickle.load(f_cp)
        with open(op_address, "rb") as f_op:
            op = pickle.load(f_op)
        return data, time, cp, op

    def fill_time(self):  # 按照time_all，将所有data的未记录时间补零
        data_name = self.name + "data_all.npy"
        data_address = osp.join(self.root, data_name)
        cp_name = self.name + "cp_all.npy"
        cp_address = osp.join(self.root, cp_name)
        op_name = self.name + "op_all.npy"
        op_address = osp.join(self.root, op_name)
        if (osp.exists(data_address)) & (osp.exists(cp_address)) & (osp.exists(op_address)):  # 若已存在，直接读取文件
            data_fill = np.load(data_address)
            cp_fill = np.load(cp_address)
            op_fill = np.load(op_address)
        else:  # 若文件不存在，计算并保存
            num_wells = len(self.well_names)
            time = np.array(self.time_all_list)  # 所有井的采集时间
            data_fill, cp_fill, op_fill = [], [], []
            for i in range(num_wells):
                data_one = self.data[i]
                cp_one = self.cp[i]
                op_one = self.op[i]
                time_one = self.time[i]  # 某口油井的所有采集时间
                data_fill_one = np.zeros(shape=time.shape[0])
                cp_fill_one = np.zeros(shape=time.shape[0])
                op_fill_one = np.zeros(shape=time.shape[0])
                for j in range(data_one.shape[0]):
                    time_one_j = time_one[j]  # 该油井的某一个采集时间
                    index = int(np.argwhere(time_one == time_one_j))
                    data_fill_one[index] = data_one[index]
                    cp_fill_one[index] = cp_one[index]
                    op_fill_one[index] = op_one[index]
                data_fill_one = data_fill_one.reshape(1, -1)
                cp_fill_one = cp_fill_one.reshape(1, -1)
                op_fill_one = op_fill_one.reshape(1, -1)
                if i != 0:
                    data_fill = np.vstack([data_fill, data_fill_one])
                    cp_fill = np.vstack([cp_fill, cp_fill_one])
                    op_fill = np.vstack([op_fill, op_fill_one])
                else:
                    data_fill = data_fill_one
                    cp_fill = cp_fill_one
                    op_fill = op_fill_one
            np.save(data_address, data_fill)  # 保存文件
            np.save(cp_address, cp_fill)
            np.save(op_address, op_fill)
        return data_fill, cp_fill, op_fill

    def train_test_split_fill_time(self, ratio=0.75):  # 在fill_time()基础上划分训练集与测试集
        data_all = self.fill_time()
        num = data_all.shape[1]
        num_train = int(ratio * num)
        data_train = data_all[:, :num_train]
        data_test = data_all[:, num_train:]
        return data_train, data_test

    def plot_zero_ratio(self, plot=True):  # 计算并绘制每个采集时间上，零值（没产量？）的比例
        zero_ratio_all = []
        for i in range(self.num_time):
            data_one = self.data_all[:, i]
            num_zero = np.sum(data_one == 0.)
            zero_ratio = num_zero / self.num_well
            zero_ratio_all.append(zero_ratio)
        zero_ratio_all = np.array(zero_ratio_all)
        if plot:
            plot_array(a=zero_ratio_all, x_name="well index", y_name="percentage")
        return zero_ratio_all

    def remain_xy_range(self, x_range, y_range):  # 仅保留一定范围内的节点及其数据
        x_min, x_max, y_min, y_max = x_range[0], x_range[1], y_range[0], y_range[1]
        remain_index = []
        for i in range(self.num_well):
            xy_one = self.xy[i, :]  # 一个井的坐标
            x_one, y_one = xy_one[0], xy_one[1]
            if (x_one > x_min) & (x_one < x_max) & (y_one > y_min) & (y_one < y_max):
                remain_index.append(i)
        xy_remain = self.xy[remain_index, :]
        data_remain = self.data_all[remain_index, :]
        return xy_remain, data_remain


# 判断某个sheet在sheet_group中的编号
def index_in_sheet_group(sheet_group, length_group, index):
    num_groups = len(sheet_group)
    row = 0
    for i in range(num_groups):
        length_one_group = length_group[i]
        if index > (length_one_group - 1):
            index = index - length_one_group
            row = row + 1
        else:
            col = index[0, 0]
            break
    return row, col


# 获得时间列表中所有的时间
def get_all_time(time):
    num_groups = len(time)
    time_list, time_arr_all, time_value_all = [], [], []
    for i in range(num_groups):
        time_list_one = time[i]
        time_one_all = []
        for j in range(time_list_one.shape[0]):
            time_one = time_list_one[j]
            time_one_all.append(time_one)
        time_list = time_list + time_one_all
        time_list = list(set(time_list))

    # 将列表的字符串变成数组形式
    num_times = len(time_list)
    for i in range(num_times):
        time_one = time_list[i]
        k, time_arr, j, value, time_value = 0, np.zeros(shape=3), 0, 0, 0
        length = len(time_one)  # 日期字符串的长度
        while True:
            a = time_one[k]  # 取出字符串中的元素
            if a == "/":  # 元素不为数字
                time_arr[j] = value
                if value < 10:
                    time_value = str(time_value) + "0" + str(value)
                else:
                    time_value = str(time_value) + str(value)
                time_value = int(time_value)
                j = j + 1
                value = 0
            else:  # 元素为数字
                value = value * 10 + int(a)
            k = k + 1
            if k >= length:  # 读取完字符串，跳出循环
                time_arr[j] = value
                if value < 10:
                    time_value = str(time_value) + "0" + str(value)
                else:
                    time_value = str(time_value) + str(value)
                time_value = int(time_value)
                break
        time_arr = time_arr.reshape(1, time_arr.shape[0])
        if i != 0:
            time_arr_all = np.vstack([time_arr_all, time_arr])
        else:
            time_arr_all = time_arr
        time_value_all.append(time_value)

    time_arr = time_arr_all.astype(np.int)
    time_value = np.array(time_value_all)
    # 对日期进行排序
    index = np.argsort(time_value)
    time_arr_sort = time_arr[index, :]
    time_list_sort = []
    for i in range(index.shape[0]):
        index_one = index[i]
        time_list_sort.append(time_list[index_one])

    return time_list_sort, time_arr_sort


# class StaticData(InMemoryDataset):
#     def __init__(self, name, well_info, root="tf_data", transform=None, pre_transform=None, pkl="new_SD"):
#         self.name = name
#         self.well_info = well_info
#         self.pkl = pkl              # pkl文件存放的位置
#         super(StaticData, self).__init__(root, transform, pre_transform)
#
#     @property
#     def raw_dir(self):
#         raw_address = self.pkl
#         return raw_address
#
#     @property
#     def processed_dir(self):
#         processed_address = osp.join(self.root, self.name, "processed")
#         return processed_address
#
#     @property
#     def raw_file_names(self):
#         data_file = self.name + "data.pkl"
#         time_file = self.name + "time.pkl"
#         names = [data_file, time_file]
#         return names
#
#     @property
#     def processed_file_names(self):
#         return 'data.pt'
#
#     def process(self):
#         data = process_well_info(self.well_info)
#         data = data if self.pre_transform is None else self.pre_transform(data)
#         torch.save(self.collate([data]), self.processed_paths[0])
#
#
# def process_well_info(well_info):
#     well_info = well_info
#     data_all = well_info.fill_time()
#     time_all = well_info.time_all_list()
#
#     return 0


# 油井静态数据 and 相关信息
class NewWellInfo(object):
    def __init__(self, name, root="../dataset/static_data/new_SD"):
        self.root = root  # 相关数据文件的存放路径
        self.name = name  # 文件名
        self.info = pd.read_csv(osp.join(self.root, self.name + ".csv"), index_col=0)
        self.update_index()
        self.data, self.time, self.cp, self.op = self.pickle_load()
        self.well_name = self.info.loc[:, "井名"].values.reshape(-1)
        self.depth = self.info.loc[:, "校深"].values.reshape(-1)
        self.res = self.info.loc[:, "电阻率"].values.reshape(-1)
        self.lag = self.info.loc[:, "声波时差"].values.reshape(-1)
        self.density = self.info.loc[:, "岩石密度"].values.reshape(-1)
        self.cni = self.info.loc[:, "补偿中子"].values.reshape(-1)
        self.mud = self.info.loc[:, "泥质含量"].values.reshape(-1)
        self.porosity = self.info.loc[:, "总孔隙度"].values.reshape(-1)
        self.perm = self.info.loc[:, "渗透率"].values.reshape(-1)
        self.hyd = self.info.loc[:, "含油气饱和度"].values.reshape(-1)
        self.xy = self.info.loc[:, ["井口纵坐标X", "井口横坐标Y"]].values
        self.data_fill_time = None  # 对非采集时间的日产气量补零
        self.cp_fill_time = None  # 对非采集时间的套压补零
        self.op_fill_time = None  # 对非采集时间的油压补零
        self.time_all_list = None  # 所有采集时间的list格式
        self.time_all_arr = None  # 所有采集时间的array格式
        self.time_all_list_month = None  # 所有采集时间的list格式，仅考虑年、月
        self.time_all_arr_month = None  # 所有采集时间的array格式，仅考虑年、月

    def pickle_load(self):  # 读取data和time，pkl格式
        data_address = osp.join(self.root, self.name + "data.pkl")
        time_address = osp.join(self.root, self.name + "time.pkl")
        cp_address = osp.join(self.root, self.name + "cp.pkl")
        op_address = osp.join(self.root, self.name + "op.pkl")
        with open(data_address, "rb") as f_data:
            data = pickle.load(f_data)
        with open(time_address, "rb") as f_time:
            time = pickle.load(f_time)
        with open(cp_address, "rb") as f_cp:
            cp = pickle.load(f_cp)
        with open(op_address, "rb") as f_op:
            op = pickle.load(f_op)
        return data, time, cp, op

    def remain_by_xy_range(self, x_range, y_range):  # 仅保留一定范围内的节点及其数据
        x_min, x_max, y_min, y_max = x_range[0], x_range[1], y_range[0], y_range[1]
        remain_index = []
        num_well = self.xy.shape[0]  # 井的数量
        for i in range(num_well):
            xy_one = self.xy[i, :]  # 一个井的坐标
            x_one, y_one = xy_one[0], xy_one[1]
            if (x_one > x_min) & (x_one < x_max) & (y_one > y_min) & (y_one < y_max):
                remain_index.append(i)  # 保留在范围内的节点索引
        self.delete_well_index(index=remain_index)  # 保留数据
        self.update_index()
        return None

    def delete_well_index(self, index):  # 仅保留index内的节点，index为节点索引
        self.cni = self.cni[index]
        self.density = self.density[index]
        self.depth = self.depth[index]
        self.hyd = self.hyd[index]
        self.info = self.info.iloc[index, :]
        self.lag = self.lag[index]
        self.mud = self.mud[index]
        self.perm = self.perm[index]
        self.porosity = self.porosity[index]
        self.res = self.res[index]
        self.well_name = self.well_name[index]
        self.xy = self.xy[index, :]
        data, time = [], []
        for i in range(len(index)):
            index_one = index[i]
            data_one = self.data[index_one]
            time_one = self.time[index_one]
            data.append(data_one)
            time.append(time_one)
        self.data = data
        self.time = time
        return None

    def plot_data(self, *args):  # 绘制某个静态数据，输入为["cni", "perm"]类型
        data_name_all = args
        for i in range(len(data_name_all)):
            data_name = data_name_all[i]
            data = self.select_data(data_name=data_name)
            plot_array(a=data, title=data_name)
        return None

    def remain_by_data(self, data_name, v_max=None, v_min=None):  # 根据数据，选择保留节点的方式
        data = self.select_data(data_name=data_name)
        if v_max is not None:
            index = np.argwhere(data < v_max).reshape(-1).tolist()
            self.delete_well_index(index=index)
        if v_min is not None:
            index = np.argwhere(data > v_min).reshape(-1).tolist()
            self.delete_well_index(index=index)
        self.update_index()
        return None

    def select_data(self, data_name):  # 挑选某个类型的静态数据
        if data_name == "cni":
            data = self.cni
        elif data_name == "density":
            data = self.density
        elif data_name == "depth":
            data = self.depth
        elif data_name == "hyd":
            data = self.hyd
        elif data_name == "lag":
            data = self.lag
        elif data_name == "mud":
            data = self.mud
        elif data_name == "perm":
            data = self.perm
        elif data_name == "porosity":
            data = self.porosity
        elif data_name == "res":
            data = self.res
        return data

    def update_index(self):  # 让节点的索引为[0, 1, ..., N-1]，N为节点数量
        info_index = np.arange(self.info.shape[0]).tolist()
        self.info.index = info_index
        return None

    def get_graph(self, gtype, k=None, s=None, name=None, sigma=None):  # 创建图结构
        if gtype == "knn":  # k近邻图
            if (k is None) or (s is None):
                raise ValueError("if gtype is 'knn', k or s can't be None")
            graph = KnnGraph(xy=self.xy, k=k, s=s)
        elif gtype == "MaxDisGraph":
            if (name is None) or (sigma is None) or (s is None):
                raise ValueError("if gtype is 'MaxDisGraph', name or sigma or s can't be None")
            graph = MaxDisGraph(xy=self.xy, name=name, sigma=sigma, s=s)
        elif gtype == "delaunay":
            graph = DLNGraph(xy=self.xy)
        else:
            raise TypeError("Unknown gtype, but got {}".format(gtype))
        return graph

    def select_data_list(self, name_list):  # 根据静态数据的类型列表，挑选多个静态数据
        num_name = len(name_list)  # 静态数据的类型的数量
        data_all = []
        for i in range(num_name):
            name_one = name_list[i]
            data_one = self.select_data(data_name=name_one).reshape(-1, 1)
            if i != 0:
                data_all = np.hstack([data_all, data_one])
            else:
                data_all = data_one
        return data_all

    def fill_time(self, name):  # 将所有data的未采集时间补零，并记录下所有的采集时间
        data_name = name + "_data_fill_time.npy"
        data_address = osp.join(self.root, data_name)
        cp_name = name + "_cp_fill_time.npy"
        cp_address = osp.join(self.root, cp_name)
        op_name = name + "_op_fill_time.npy"
        op_address = osp.join(self.root, op_name)

        list_name, arr_name = name + "_time_all_list.npy", name + "_time_all_arr.npy"
        list_address, arr_address = osp.join(self.root, list_name), osp.join(self.root, arr_name)

        if osp.exists(data_address) & osp.exists(cp_address) & osp.exists(op_address):  # 若文件已保存
            self.data_fill_time = np.load(data_address)
            self.cp_fill_time = np.load(cp_address)
            self.op_fill_time = np.load(op_address)
            self.time_all_list = np.load(list_address)  # 读取时，list已变成array格式
            self.time_all_arr = np.load(arr_address)
        else:  # 若文件尚未保存
            used = WellInfo(name="all", fish="all_fish", data="alldata", time="alltime", cp="allcp", op="allop")
            time_all_list, time_all_arr = used.time_all_list, used.time_all_arr  # 所有采集时间，list格式和array格式
            np.save(list_address, time_all_list)  # 保存所有采集时间的list格式
            np.save(arr_address, time_all_arr)  # 保存所有采集时间的array格式
            data_fill_time, all_well_names = used.data_all, used.well_names
            cp_fill_time, op_fill_time = used.cp_all, used.op_all
            well_names = self.well_name
            num_well = well_names.shape[0]  # 考虑的井的数量
            data_fill_time_all, cp_fill_time_all, op_fill_time_all = [], [], []
            for i in range(num_well):
                well_name_one = well_names[i]  # 某个井的名字
                for j in range(len(all_well_names)):  # 找到该井在all_well_names中的位置
                    well_name_j = all_well_names[j]
                    if well_name_j == well_name_one:
                        break
                data_fill_time_one = data_fill_time[j, :].reshape(1, -1)  # 提取出该井的data已补零格式
                cp_fill_time_one = cp_fill_time[j, :].reshape(1, -1)
                op_fill_time_one = op_fill_time[j, :].reshape(1, -1)
                if i != 0:
                    data_fill_time_all = np.vstack([data_fill_time_all, data_fill_time_one])
                    cp_fill_time_all = np.vstack([cp_fill_time_all, cp_fill_time_one])
                    op_fill_time_all = np.vstack([op_fill_time_all, op_fill_time_one])
                else:
                    data_fill_time_all = data_fill_time_one
                    cp_fill_time_all = cp_fill_time_one
                    op_fill_time_all = op_fill_time_one
            np.save(data_address, data_fill_time_all)
            np.save(cp_address, cp_fill_time_all)
            np.save(op_address, op_fill_time_all)
            self.data_fill_time = data_fill_time_all
            self.cp_fill_time = cp_fill_time_all
            self.op_fill_time = op_fill_time_all
        return self.data_fill_time, self.cp_fill_time, self.op_fill_time, self.time_all_list, self.time_all_arr

    def merge_data(self, name, style="month"):  # 按照某种方式合并日产气量
        if style == "month":  # 将一个月的日产气量加起来
            if self.data_fill_time is not None:
                data_fill_time, time_all_list, time_all_arr = self.data_fill_time, self.time_all_list, self.time_all_arr
            else:
                data_fill_time, time_all_list, time_all_arr = self.fill_time(name)
            num_all_time = time_all_arr.shape[0]  # 所有采集时间的数量
            month_last, day_last, data_month_all, time_month_all = 0, 0, None, []
            time_all_arr_month, time_all_list_month = None, []
            for i in range(num_all_time):
                data_one = data_fill_time[:, i].reshape(-1, 1)  # 某一采集时间，所有井的日产气量
                time_one_arr = time_all_arr[i, :]  # 某一采集时间，年月日为数组形式
                year, month, day = int(time_one_arr[0]), int(time_one_arr[1]), int(time_one_arr[2])  # 年、月、日
                if (month != month_last) | (i == num_all_time - 1):  # 采集时间与上一个采集时间不是同一个月，或运行到最后一次
                    if i != 0:  # 不是第一次运行，拼接上一个月份的日产气量
                        if data_month_all is None:  # 是第一个月
                            data_month_all = data_month
                        else:  # 不是第一个月
                            data_month_all = np.hstack([data_month_all, data_month])
                    data_month = data_one  # 当前月份第一天的日产气量
                    month_last = month

                    time_one_arr_month = np.array([year, month]).reshape(1, -1)  # 仅考虑年月，array格式
                    if time_all_arr_month is None:
                        time_all_arr_month = time_one_arr_month
                    else:
                        time_all_arr_month = np.vstack([time_all_arr_month, time_one_arr_month])

                    time_one_list_month = str(year) + "/" + str(month)  # 仅考虑年月，list格式存储为字符串
                    time_all_list_month.append(time_one_list_month)

                else:  # 采集时间与上一个采集时间为同一个月
                    data_month = data_month + data_one
                self.time_all_list_month = time_all_list_month
                self.time_all_arr_month = time_all_arr_month
        else:
            raise TypeError("style must be 'month', but got {}".format(style))
        return data_month_all, time_all_list_month, time_all_arr_month

    def choice_dynamic_data(self, well_list, rz_style, all_well, features):  # 由名称，挑选油井的日产气量or套压or油压
        if all_well:  # 考虑所有的油井
            if rz_style:
                raise ValueError("If all_well is True, rz_style must be None, otherwise the minimum length of"
                                 " data is too small")
            well_list = self.well_name
        num_choice = len(well_list)  # 挑选的油井的数量
        data_all_well, cp_all_well, op_all_well = None, None, None  # 初始化动态数据
        length_all = []  # 记录下除零后，各个油井的日产气量的长度
        for i in range(num_choice):
            well_one = well_list[i]  # 读取一口油井
            well_location = np.argwhere(self.well_name == well_one)  # 该井在数据集中的位置
            if well_location.shape[0] == 0:  # 若该井在数据集中不存在，报错
                raise ValueError("{} is not in the {}".format(well_one, self.name))
            else:
                well_location = well_location.reshape(-1)[0]
                data_one_well = self.data_fill_time[well_location, :].reshape(-1, 1)  # 该井在所有采集时间上的日产气量
                cp_one_well = self.cp_fill_time[well_location, :].reshape(-1, 1)  # 该井在所有采集时间上的套压
                op_one_well = self.op_fill_time[well_location, :].reshape(-1, 1)  # 该井在所有采集时间上的油压
            if (data_all_well is None) & (cp_all_well is None) & (op_all_well is None):  # 拼接所有井的动态数据
                data_all_well = data_one_well
                cp_all_well = cp_one_well
                op_all_well = op_one_well
            else:
                data_all_well = np.hstack([data_all_well, data_one_well])
                cp_all_well = np.hstack([cp_all_well, cp_one_well])
                op_all_well = np.hstack([op_all_well, op_one_well])

        if rz_style is not None:  # 需要基于日产气量数据，进行除零操作
            data_all, cp_all, op_all = None, None, None  # 除零后，各个油井日产气量、套压的拼接
            if rz_style == "rz_one":  # 各油井的日产气量除零后，拼接，长度为除零后某井日产气量的最小长度
                length_min = float('inf')  # 初始化最小长度
                for i in range(num_choice):
                    data_one, cp_one, op_one = data_all_well[:, i], cp_all_well[:, i], op_all_well[:, i]
                    index = np.argwhere(data_one).reshape(-1)  # 零值在日产气量中的位置
                    data_one, cp_one, op_one = data_one[index], cp_one[index], op_one[index]  # 除去各自的零值
                    length_one = data_one.shape[0]  # 除零后，日产气量的长度
                    length_all.append(length_one)
                    if length_one < length_min:
                        length_min = length_one  # 选择所有井中日产气量的最小长度
                    data_one = data_one[:length_min].reshape(-1, 1)  # 仅保留前n个值，n为最小长度
                    cp_one = cp_one[:length_min].reshape(-1, 1)
                    op_one = op_one[:length_min].reshape(-1, 1)
                    if data_all is None:  # 拼接除零后的日产气量
                        data_all, cp_all, op_all = data_one, cp_one, op_one
                    else:
                        if data_all.shape[0] > length_min:  # 若当前总日产气量的长度大于最小长度n，仅保留前n个值
                            data_all = data_all[:length_min, :]
                            cp_all = cp_all[:length_min, :]
                            op_all = op_all[:length_min, :]
                        data_all = np.hstack([data_all, data_one])
                        cp_all = np.hstack([cp_all, cp_one])
                        op_all = np.hstack([op_all, op_one])
                data_all_well, cp_all_well, op_all_well = data_all, cp_all, op_all
        if features == ["日产气量(10⁴m³/d)"]:  # 使用的数据集仅考虑日产气量
            return data_all_well, length_all
        elif features == ["日产气量(10⁴m³/d)", "套压(MPa)"]:
            data_all_well = np.expand_dims(data_all_well, axis=2)
            cp_all_well = np.expand_dims(cp_all_well, axis=2)
            data_used = np.concatenate((data_all_well, cp_all_well), axis=2)
        elif features == ["日产气量(10⁴m³/d)", "油压(MPa)"]:
            data_all_well = np.expand_dims(data_all_well, axis=2)
            op_all_well = np.expand_dims(op_all_well, axis=2)
            data_used = np.concatenate((data_all_well, op_all_well), axis=2)
        elif features == ["日产气量(10⁴m³/d)", "套压(MPa)", "油压(MPa)"]:
            data_all_well = np.expand_dims(data_all_well, axis=2)
            cp_all_well = np.expand_dims(cp_all_well, axis=2)
            op_all_well = np.expand_dims(op_all_well, axis=2)
            data_used = np.concatenate((data_all_well, cp_all_well, op_all_well), axis=2)
        return data_used, length_all

    def get_well_data(self, well, rz=False):  # 找到某口油井的日产气量
        index = np.argwhere(self.well_name == well)
        if index.shape[0] == 0:  # 未找到该井
            raise ValueError("{} is not in the {}".format(well, self.name))
        else:
            index = index.reshape(-1)[0]
        well_data = self.data[index]
        if rz:  # 选择对日产气量除零
            index = np.argwhere(well_data).reshape(-1)
            well_data = well_data[index]
        return well_data

    def get_well_list_data(self, well_list, rz=False):  # 找到一些油井的日产气量
        num_well = len(well_list)  # 油井的数量
        data_all = []  # 日产气量list
        for i in range(num_well):
            well_one = well_list[i]  # 单口油井
            data_one = self.get_well_data(well_one, rz)  # 单口油井的日产气量
            data_all.append(data_one)
        return data_all

    def select_well_in_new(self, well_list):  # 输入list，挑选出位于new中的油井
        num = len(well_list)  # 油井数量
        well_list_selected = []
        for i in range(num):
            well_name_one = well_list[i]
            index = np.argwhere(self.well_name == well_name_one)
            if index.shape[0] == 0:  # 该油井不在new中
                continue
            else:  # 该油井在new中
                well_list_selected.append(well_name_one)
        return well_list_selected

    def eval_regress(self, well, regress, ratio, x_length, y_length, title, rz=False, score=True, rmse=False,
                     fig_size=(12, 8), alpha=0.5):
        """
        对单口油井的日产气量，构建回归器，进行机器学习测试
        :param well: 油井名称，类型str
        :param regress: 回归器
        :param ratio: 训练集所占比例，类型float
        :param x_length: 数据序列长度，类型int
        :param y_length: 标签序列长度，类型int
        :param title: 绘图标题，类型str
        :param rz: 是否除零，类型bool
        :param fig_size: 绘图大小，类型tuple
        :param alpha: 曲线透明度，类型float
        :param score: 标题是否添加决定系数score，类型bool
        :param rmse: 计算并在标题中添加均方根误差，类型bool
        :return:
        """
        data = self.get_well_data(well)
        if rz:
            index = np.argwhere(data).reshape(-1)
            data = data[index]
        num = data.shape[0]
        num_train = int(ratio * num)
        data_train, data_test = data[:num_train], data[num_train:]
        x_train, y_train = get_xy(data_train, x_length, y_length, style="arr")
        x_test, y_test = get_xy(data_test, x_length, y_length, style="arr")
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        regress.fit(x_train, y_train)
        y_train_pred = regress.predict(x_train)
        y_test_pred = regress.predict(x_test)
        train_score = regress.score(x_train, y_train)
        test_score = regress.score(x_test, y_test)

        plt.figure(figsize=fig_size)
        if score:  # 标题添加决定系数score
            title = title + "\nTrain Score={:.5f}, Test Score={:.5f}".format(train_score, test_score)
        if rmse:  # 计算并在标题添加均方根误差
            x_train_tensor, y_train_tensor = torch.from_numpy(x_train), torch.from_numpy(y_train)
            x_test_tensor, y_test_tensor = torch.from_numpy(x_test), torch.from_numpy(y_test)
            criterion = torch.nn.MSELoss()
            loss_train = criterion(x_train_tensor, y_train_tensor)
            loss_test = criterion(x_test_tensor, y_test_tensor)
            title = title + "\nTrain Loss={:.5f}， Test Loss={:.5f}".format(loss_train, loss_test)

        plt.title(well + ", " + title, fontsize=20)
        range_train = np.arange(y_train.shape[0])  # 训练集绘图的横坐标取值
        range_test = np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0])  # 测试集绘图的横坐标取值
        plt.plot(range_train, y_train, label="train", alpha=alpha)
        plt.plot(range_train, y_train_pred, label="train predict", alpha=alpha)
        plt.plot(range_test, y_test, label="test", alpha=alpha)
        plt.plot(range_test, y_test_pred, label="test predict", alpha=alpha)
        plt.legend(fontsize=15)
        plt.ylabel("Daily Production", fontsize=20)
        plt.xlabel("Date", fontsize=20)
        return train_score, test_score

    def total_variation(self, adm, style="data", norm=False, time="one"):  # 对每个采集时刻/所有采集时刻的日产气量，计算全变差
        if style == "data":  # 在所有采集时刻上的日产气量
            if self.data_fill_time is None:
                data_fill_time, _, _, _, _ = self.fill_time(name="new")
            else:
                data_fill_time = self.data_fill_time
            x = data_fill_time
            num_time = data_fill_time.shape[1]  # 采集时刻的数量
        else:
            raise TypeError("style must be 'data', but got {}".format(style))
        dm = get_degree_matrix(adm)  # 计算度矩阵
        lam = get_laplacian_matrix(adm, dm, norm)  # 拉普拉斯矩阵
        if time == "one":
            tv = []
            for i in range(num_time):  # 计算每一个采集时刻上的total variation
                x_one = x[:, i]
                x_one_row, x_one_col = x_one.reshape(1, -1), x_one.reshape(-1, 1)
                tv_one = float(np.dot(np.dot(x_one_row, lam), x_one_col))
                tv.append(tv_one)
            return tv
        elif time == "all":
            tv = np.trace(np.dot(np.dot(x.T, lam), x))
            return tv
        else:
            raise TypeError("time must be 'one' or 'all', but got {}".format(time))

    def sum_data_fill_time(self, node_index=None):  # 计算不同的油井，所有日产气量的和
        if self.data_fill_time is None:
            data_fill_time, _, _, _, _ = self.fill_time(name="new")
        else:
            data_fill_time = self.data_fill_time
        if node_index is None:  # 考虑所有油井
            node_index = np.arange(data_fill_time.shape[0])
        data = data_fill_time[node_index, :]  # 选出这些油井的日产气量（在所有采集时间）
        data_sum = np.sum(data, axis=1)
        return data_sum

    def select_static_data_well_list(self, well_list, static_style=None):  # 挑选一些油井的静态数据
        num = len(well_list)  # 油井的数量
        if static_style is None:  # 若为None，默认提取所有静态特征
            static_style = ["cni", "density", "depth", "hyd", "lag", "mud", "perm", "porosity", "res"]
        static_data = self.select_data_list(static_style)  # 挑选这些静态特征
        static_data_well_all = None
        for i in range(num):
            well_one = well_list[i]
            index = np.argwhere(self.well_name == well_one)
            if index.shape[0] == 0:
                raise ValueError("{} is not in the {}".format(well_one, self.name))
            else:
                index = index.reshape(-1)[0]
                static_data_well_one = static_data[index, :].reshape(1, -1)  # 提该油井的静态特征
                if static_data_well_all is None:
                    static_data_well_all = static_data_well_one
                else:
                    static_data_well_all = np.vstack([static_data_well_all, static_data_well_one])
        return static_data_well_all

    def plot_well_in_block(self, block_name, block_well_name, legend=False, font_size_le=8):
        """
        按照不同的区块，绘制油井
        :param block_name: 所有区块的名称数组，类型array
        :param block_well_name: 各个区块包含的油井名称列表，类型list
        :param legend: 是否显示标注，类型bool
        :param font_size_le: 标注字体大小，类型int
        :return:
        """
        num_blocks = block_name.shape[0]  # 区块数量
        xy_all = []  # 所有区块的油井坐标
        for i in range(num_blocks):
            block_one = block_name[i]  # 当前区块
            block_well_one = block_well_name[i]  # 当前区块的油井
            num_well = len(block_well_one)  # 当前区块的油井数量
            xy = None
            for j in range(num_well):
                well_one = block_well_one[j]  # 当前油井
                index = np.argwhere(self.well_name == well_one)
                if index.shape[0] == 0:  # 该油井在NewWellInfo中不存在
                    continue
                else:
                    index = index.reshape(-1)[0]  # 当前油井在NewWellInfo中的索引
                    xy_one = self.xy[index, :].reshape(1, -1)
                if xy is None:
                    xy = xy_one
                else:
                    xy = np.vstack([xy, xy_one])  # 一个区块内，油井的坐标
            xy_all.append(xy)

        plt.figure(figsize=(12, 8))
        for i in range(len(xy_all)):
            plt.scatter(xy_all[i][:, 0], xy_all[i][:, 1], label="{}".format(block_name[i]))
        if legend:
            plt.legend(fontsize=font_size_le)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        return xy_all


# 判断一个节点集合中，相连的节点
def graph_cluster(nodes, edge_index):
    edge_index = edge_index.detach().cpu().numpy()  # 边索引
    u, v = edge_index[0, :], edge_index[1, :]
    clu_all, j_all = [], []
    while True:
        clu_one = []  # 初始化簇集合（相邻点）
        node = nodes[0]  # 当前节点
        clu_one.append(node)  # 将当前节点，加入簇集合
        clu_one_else, j = [], 1
        while True:
            nei_index = np.argwhere(u == node)  # 邻点索引
            if nei_index.shape[0] == 0:  # 如果当前节点，没有邻点
                pass
            else:  # 如果当前节点有相邻点，将相邻节点加入簇集合
                nei_index = nei_index.reshape(-1)
                nei = v[nei_index].tolist()  # 邻点
                clu_one = clu_one + nei  # 更新簇集合，用于最后输出
                clu_one_else = clu_one_else + nei  # 更新簇集合，用于每次除去当前节点
                clu_one = list(set(clu_one))  # 除去重复节点
            clu_one_else = list(set(clu_one_else) - set([node]))  # 除去当前节点
            nodes, edge_index = delete_graph_nodes_edge_index(nodes, edge_index, node)  # 从图结构中除去该节点信息
            u, v = edge_index[0, :], edge_index[1, :]
            if len(clu_one_else) == 0:  # 簇集合中所有节点都已考虑
                break
            else:  # 簇集合中还有未考虑的节点
                node = clu_one_else[0]
                j = j + 1
        clu_all.append(clu_one)
        j_all.append(j)
        if nodes.shape[0] == 0:  # 所有节点全部考虑
            break
    return clu_all


def delete_graph_nodes_edge_index(nodes, edge_index, i):
    """
    从一个图结构（节点集合、边索引）中，除去一个节点的信息
    :param nodes: 当前图结构的节点集合，类型array
    :param edge_index: 当前图结构的边索引，类型array
    :param i: 当前节点，类型int
    :return: 节点集合、边索引
    """
    nodes_remain = list(set(nodes) - set([i]))  # 保留的节点集合
    nodes_remain = np.array(nodes_remain)
    u, v = edge_index[0, :], edge_index[1, :]
    u_index, v_index = np.argwhere(u == i), np.argwhere(v == i)
    if (u_index.shape[0] == 0) & (v_index.shape[0] == 0):
        edge_index_remain = edge_index
    elif (u_index.shape[0] != 0) & (v_index.shape[0] != 0):
        u_index, v_index = u_index.reshape(-1), v_index.reshape(-1)
        index = np.array(u_index.tolist() + v_index.tolist())
        u_remain, v_remain = np.delete(u, index), np.delete(v, index)
        u_remain, v_remain = u_remain.reshape(1, -1), v_remain.reshape(1, -1)
        edge_index_remain = np.vstack([u_remain, v_remain])
    else:
        raise ValueError("the graph is not a undirected graph")
    return nodes_remain, edge_index_remain


# 根据节点的簇集合，利用坐标距离，完善edge_index
def improve_edge_index(edge_index, nodes_cluster, nodes, xy):
    edge_index = edge_index.detach().cpu().numpy()
    edge_index_improve = edge_index
    num_clu = len(nodes_cluster)
    for i in range(num_clu):
        clu_one = nodes_cluster[i]  # 簇集合内的节点
        clu_one_else = list(set(list(nodes)) - set(clu_one))  # 簇集合外的节点
        a = xy[clu_one, :].astype(np.float32)  # 簇集合内的节点的坐标
        batch_a = torch.tensor([0] * len(clu_one))
        b = xy[clu_one_else, :].astype(np.float32)  # 簇集合外的节点的坐标
        batch_b = torch.tensor([0] * len(clu_one_else))
        a, b = torch.from_numpy(a), torch.from_numpy(b)

        assign_index = knn(b, a, 1, batch_b, batch_a).detach().cpu().numpy()  # 距离每个簇内节点最近的簇外节点坐标
        nearest_node_else = assign_index[1, :]
        counts = np.bincount(nearest_node_else)
        node_out = clu_one_else[np.argmax(counts)]  # 取众数，认为该节点是距离当前节点最近的簇外节点

        node_out_xy = xy[node_out, :].astype(np.float32).reshape(1, 2)  # 该节点的坐标
        c, batch_c = torch.from_numpy(node_out_xy), torch.tensor([0])
        assign_index_c = knn(a, c, 1, batch_a, batch_c).detach().cpu().numpy()
        node_in = clu_one[assign_index_c.reshape(-1)[1]]  # 距离node_out最近的簇内节点

        edge_index_add = np.array([[node_in, node_out], [node_out, node_in]])
        edge_index_improve = np.hstack([edge_index_improve, edge_index_add])  # 更新边索引
    edge_index_improve = torch.from_numpy(edge_index_improve)

    return edge_index_improve


# 将邻接矩阵的类型由array转化为edge_index
def tran_adm_to_edge_index(adm):
    if isinstance(adm, np.ndarray):  # array数组类型
        u, v = np.nonzero(adm)
        num_edges = u.shape[0]  # 边的数量
        edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])  # 边索引
        edge_weight = np.zeros(shape=u.shape)  # 初始化边权重
        for i in range(num_edges):
            edge_weight_one = adm[u[i], v[i]]
            edge_weight[i] = edge_weight_one
        edge_index = torch.from_numpy(edge_index.astype(np.int64))
        edge_weight = torch.from_numpy(edge_weight.astype(np.float32))
        return edge_index, edge_weight
    elif torch.is_tensor(adm):  # pytorch张量类型
        u, v = torch.nonzero(adm, as_tuple=True)
        num_edges = u.shape[0]  # 边的数量
        edge_index = torch.cat((u.view(1, -1), v.view(1, -1)))
        edge_weight = torch.zeros(size=u.shape)
        for i in range(num_edges):
            edge_weight[i] = adm[u[i], v[i]]
        return edge_index, edge_weight


# 将临界矩阵的类型由edge_index转化为array
def tran_edge_index_to_adm(edge_index, num_nodes_truth=None):
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index_r1, edge_index_r2 = edge_index[0, :].reshape(-1), edge_index[1, :].reshape(-1)

    # 节点的数量
    edge_index_r1_r2 = np.hstack([edge_index_r1, edge_index_r2])
    if num_nodes_truth is not None:  # 当图不是连接图时（存在孤立节点时）
        num_nodes = num_nodes_truth
        nodes = np.arange(num_nodes)
    else:
        nodes = np.unique(edge_index_r1_r2)
        num_nodes = nodes.shape[0]
    adm = np.zeros(shape=(num_nodes, num_nodes))

    for i in range(num_nodes):
        node = nodes[i]
        neighbor_nodes_index = np.where(edge_index_r1 == node)[0].tolist()
        neighbor_nodes = edge_index_r2[neighbor_nodes_index].tolist()
        adm[i, neighbor_nodes] = 1.

    return adm


# 重复完善图结构，直到图满足连通性（不存在孤立节点）
def tran_graph_to_connected(nodes, graph, xy, gtype=None):
    while True:
        node_cluster = graph_cluster(nodes, graph.edge_index)
        if len(node_cluster) == 1:
            break
        edge_index = improve_edge_index(graph.edge_index, node_cluster, nodes, xy)
        adm = tran_edge_index_to_adm(edge_index)
        if gtype is None:
            graph_new = SelfGraph(xy, adm, edge_index=True)
        else:
            graph_new = SelfGraph(xy, adm, edge_index=True, gtype=gtype)
        graph = graph_new
    return graph


class InterpolationGraph(nn.Module):
    def __init__(self, edge_index, graph, p_t_s, p_t_i, p_t_a, p_w, u, u_t, dm, s_a_group, s_i, w=300, k=50,
                 use_bias=True, get_adm=True):
        super(InterpolationGraph, self).__init__()
        self.edge_index = edge_index  # 边索引
        self.num_edges = edge_index.size(1)  # 边的数量
        self.graph = graph  # 图结构
        self.num_nodes = graph.N  # 节点数量
        self.p_t_s = p_t_s  # 采样算子
        self.p_t_i = p_t_i  # 孤立节点选取算子
        self.p_t_a = p_t_a  # 冗余节点选取算子
        self.p_w = p_w  # 频域选取算子
        self.u = u  # 特征向量矩阵
        self.u_t = u_t  # GFT矩阵，（特征向量矩阵的逆矩阵）
        self.dm = dm  # 度矩阵
        self.s_a_group = s_a_group  # 冗余节点分组集合
        self.s_i = s_i  # 孤立节点集合
        self.w = w  # 认为的图信号带宽
        self.k = k  # 迭代次数
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(self.num_nodes, self.num_nodes))
        if get_adm:
            self.weight = nn.Parameter(self.get_adm())  # 使用权重，修饰邻接矩阵
            self.adm = self.weight
        else:
            self.adm = self.weight
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.num_nodes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def evaluate_cluster(self, fk, fk_s):  # 簇内赋值过程，(3-5)、(3-6)
        a = torch.mm(self.adm, fk_s.view(-1, 1)).view(-1)
        b_all = torch.zeros(size=fk.shape).to(device).view(-1)
        for j in range(len(self.s_a_group)):
            s_a_one = self.s_a_group[j]
            b = select_signal_nodes(f=a, nodes=s_a_one, style="tensor")
            b_all = b_all + b
        b_all = b_all
        gk_s = torch.mm(self.p_t_a, b_all.view(-1, 1)).view(-1)
        return gk_s

    def interpolate_cluster(self, fk_s, gk_s):  # 簇内插值过程，(3-7)
        c = np.zeros(shape=self.adm.shape).astype(np.float32)
        c = torch.from_numpy(c).to(device)
        dm_i = self.dm[self.s_i]  # 孤立节点的度
        dm_i_inv = 1 / dm_i
        dm_i_inv = torch.from_numpy(dm_i_inv.astype(np.float32)).to(device)
        c[self.s_i, self.s_i] = dm_i_inv
        p_e = torch.mm(torch.mm(c, self.adm), (fk_s + gk_s).view(-1, 1))
        fk_s_ = torch.mm(self.p_t_i, p_e.view(-1, 1)).view(-1)
        return fk_s_

    def project_space(self, fk, gk_s, fk_s, fk_s_):  # 组合信号并向带限子空间投影
        I = torch.eye(self.num_nodes).to(device)
        fk = torch.mm((I - self.p_t_s), (gk_s + fk_s_).view(-1, 1)).view(-1) + fk_s + fk.view(-1)
        fk = torch.mm(self.u_t, fk.view(-1, 1))
        fk = torch.mm(self.p_w, fk)
        fk = torch.mm(self.u, fk)
        return fk

    def forward(self, f_d):
        f_d_row, f_d_col = f_d.view(-1), f_d.view(-1, 1)
        # 初始化fk
        f_d_gft = torch.mm(self.u_t, f_d_col)
        fk = torch.mm(self.u, torch.mm(self.p_w, f_d_gft))
        for i in range(self.k):
            fk_s = f_d_row - torch.mm(self.p_t_s, fk).view(-1)
            gk_s = self.evaluate_cluster(fk, fk_s)
            fk_s_ = self.interpolate_cluster(fk_s, gk_s)
            fk = self.project_space(fk, gk_s, fk_s, fk_s_)
        output = fk.view(-1)
        return output

    def get_adm(self):  # 利用权重值，修饰邻接矩阵
        weight = self.weight.view(-1)
        adm = torch.zeros(size=(self.num_nodes, self.num_nodes))
        u, v = self.edge_index[0, :], self.edge_index[1, :]
        for i in range(self.num_edges):
            u_one, v_one = u[i], v[i]
            adm[u_one, v_one] = weight[i]
        return adm.to(device)


# 根据节点索引保留信号的一部分，其余部分置零
def select_signal_nodes(f, nodes, style="numpy"):
    if style == "numpy":
        y = np.zeros(shape=f.shape[0])
        y[nodes] = f[nodes]
        return y
    elif style == "tensor":
        y = torch.zeros(size=f.shape).to(device)
        y[nodes] = f[nodes]
        return y


# 源于Node Classification.ipynb
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, output_dim)
        self.name = "GCNConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(h, edge_index)
        # h = F.dropout(h, training=self.training)
        h = F.log_softmax(h, dim=1)
        return h


# 来源Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
class GraphConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphConv, self).__init__()
        self.conv1 = gnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = gnn.GraphConv(hidden_dim, output_dim)
        self.name = "GraphConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 来源Simple and Deep Graph Convolutional Networks
class GCN2Conv(nn.Module):
    def __init__(self, input_dim, alpha, theta, layer):
        super(GCN2Conv, self).__init__()
        self.conv1 = gnn.GCN2Conv(input_dim, alpha, theta, layer)
        self.conv2 = gnn.GCN2Conv(input_dim, alpha, theta, layer)
        self.name = "GCN2Conv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_0 = x
        x = self.conv1(x, x_0, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, x_0, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 来源Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
class ChebConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k):
        super(ChebConv, self).__init__()
        self.conv1 = gnn.ChebConv(input_dim, hidden_dim, k)
        self.conv2 = gnn.ChebConv(hidden_dim, output_dim, k)
        self.name = "ChebConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 来源Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
class ClusterGCNConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClusterGCNConv, self).__init__()
        self.conv1 = gnn.ClusterGCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.ClusterGCNConv(hidden_dim, output_dim)
        self.name = "ClusterGCNConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 来源JUST JUMP: DYNAMIC NEIGHBORHOOD AGGREGATION IN GRAPH NEURAL NETWORKS
class DNAConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNAConv, self).__init__()
        self.conv1 = gnn.DNAConv(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.name = "DNAConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.unsqueeze(1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


# 来源Attention-based Graph Neural Network for Semi-Supervised Learning
class AGNNConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConv, self).__init__()
        self.conv1 = gnn.AGNNConv()
        self.conv2 = gnn.AGNNConv()
        self.linear = nn.Linear(input_dim, output_dim)
        self.name = "AGNNConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x


# 来源Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties
class CGConv(nn.Module):
    def __init__(self, input_dim):
        super(CGConv, self).__init__()
        self.conv1 = gnn.CGConv(input_dim)
        self.conv2 = gnn.CGConv(input_dim)
        self.name = "CGConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 来源How Powerful are Graph Neural Networks
class GINConv(nn.Module):
    def __init__(self, nn, input_dim, hidden_dim, output_dim, eps=0.0):
        super(GINConv, self).__init__()
        self.nn1 = nn(input_dim, hidden_dim)
        self.nn2 = nn(hidden_dim, output_dim)
        self.conv1 = gnn.GINConv(self.nn1, eps)
        self.conv2 = gnn.GINConv(self.nn2, eps)
        self.name = "GINConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 来源DeeperGCN: All You Need to Train Deeper GCNs
class GENConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GENConv, self).__init__()
        self.conv1 = gnn.GENConv(input_dim, hidden_dim)
        self.conv2 = gnn.GENConv(hidden_dim, output_dim)
        self.name = "GENConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x


# 来源Predict then Propagate Graph Neural Networks meet Personalized PageRank
class PageRank(nn.Module):
    def __init__(self, k, alpha, input_dim, output_dim):
        super(PageRank, self).__init__()
        self.conv = gnn.GCNConv(input_dim, output_dim)
        self.page_rank1 = gnn.APPNP(K=k, alpha=alpha)
        self.page_rank2 = gnn.APPNP(K=k, alpha=alpha)
        self.name = "PageRank"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv(x, edge_index)
        h = h.relu()
        h = self.page_rank1(h, edge_index)
        h = h.relu()
        h = self.page_rank2(h, edge_index)
        h = F.log_softmax(h, dim=1)
        return h


# 来源Graph Attention Networks
class GATConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATConv, self).__init__()
        self.conv1 = gnn.GATConv(input_dim, hidden_dim)
        self.conv2 = gnn.GATConv(hidden_dim, output_dim)
        self.name = "GATConv"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# 《EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs》
class EvolveGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_edges):
        super(EvolveGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, output_dim)
        self.lstm = nn.LSTM(num_edges, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_edges)
        self.name = "EvolveGCN"

    def forward(self, data, edge_weight):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        edge_weight = edge_weight.view(1, 1, edge_index.size(1))
        edge_weight, (h0, c0) = self.lstm(edge_weight)
        edge_weight = F.relu(edge_weight)
        edge_weight = F.dropout(edge_weight, training=self.training)
        edge_weight = self.linear(edge_weight)
        edge_weight = edge_weight.view(-1)
        edge_weight = F.relu(edge_weight)
        edge_weight = F.dropout(edge_weight, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.log_softmax(h, dim=1)

        return h


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def add_data_mask(data, ratio=0.75, shuffle=False):
    """
    对单个图结构的数据生成训练集、测试集索引
    :param data: torch_geometric格式的数据
    :param ratio: 训练集所占的比例，类型float
    :param shuffle: 是否打乱数据集，类型bool
    :return: 添加了训练集索引、测试集索引的图数据
    """
    x, y = data.x, data.y  # 数据、标签
    num_nodes = x.size(0)  # 节点数量
    num_train = int(num_nodes * ratio)  # 训练集节点的数量
    if shuffle:
        nodes = np.arange(num_nodes).tolist()  # 节点索引
        train_index = np.random.choice(num_nodes, num_train, replace=False)  # 训练集节点索引
        test_index = list(set(nodes) - set(train_index.tolist()))  # 测试集节点索引（节点索引对训练集索引的补集）
        test_index = torch.from_numpy(np.array(test_index).astype(np.int64))
        train_index = torch.from_numpy(train_index.astype(np.int64))
    else:
        train_index = torch.arange(num_train, dtype=torch.long)
        test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
    train_mask = index_to_mask(train_index, size=num_nodes)
    test_mask = index_to_mask(test_index, size=num_nodes)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


# 计算并绘制标签的类别及所占比例
def target_class_ratio(target, r=True, n=False, fig_size=(12, 8)):
    if torch.is_tensor(target):  # 若为pytorch张量类型
        target = target.detach().cpu().numpy().astype(np.int64).reshape(-1, 1)
    num_target = target.shape[0]  # 标签的数量
    df = pd.DataFrame(target, columns=['target'])  # 转换为Dataframe格式
    result = df.value_counts()  # 统计标签各个类别的数量
    target_class = np.array(result.index.tolist()).reshape(-1)  # 各个类别
    num = result.values.reshape(-1)  # 各个类别的数量
    ratio = num / num_target  # 各个类别的比例
    if r:  # 绘制比例
        plt.figure(figsize=fig_size)
        plt.bar(target_class, ratio)
        plt.xlabel("Class", fontsize=20)
        plt.ylabel("Ratio", fontsize=20)
        plt.title("Label Distribution", fontsize=20)
    if n:  # 绘制数量
        plt.figure(figsize=fig_size)
        plt.bar(target_class, num)
        plt.xlabel("Class", fontsize=20)
        plt.ylabel("Num", fontsize=20)
        plt.title("Label Number", fontsize=20)

    return result


def save_tg_data(data, root, name):
    """
    保存一个torch geometric的Data类型
    :param data: torch geometric的Data类型
    :param root: 存放目录，类型str
    :param name: 保存名称，类型str
    :return:
    """
    x, edge_index, y, train_mask, test_mask = data.x, data.edge_index, data.y, data.train_mask, data.test_mask
    folder_address = osp.join(root, name)  # 文件夹路径
    if not (osp.exists(folder_address)):
        os.mkdir(folder_address)  # 创建一个空文件夹，用于存放Data的各个tensor

    # 保存的文件名
    name_x = osp.join(name + "_x.pt")
    name_edge_index = osp.join(name + "_edge_index.pt")
    name_y = osp.join(name + "_y.pt")
    name_train_mask = osp.join(name + "_train_mask.pt")
    name_test_mask = osp.join(name + "_test_mask.pt")

    # 保存文件的路径
    address_x = osp.join(folder_address, name_x)
    address_edge_index = osp.join(folder_address, name_edge_index)
    address_y = osp.join(folder_address, name_y)
    address_train_mask = osp.join(folder_address, name_train_mask)
    address_test_mask = osp.join(folder_address, name_test_mask)

    torch.save(x, address_x)
    torch.save(edge_index, address_edge_index)
    torch.save(y, address_y)
    torch.save(train_mask, address_train_mask)
    torch.save(test_mask, address_test_mask)
    return None


def load_tg_data(root, name):
    """
    读取存一个torch geometric的Data类型
    :param root: 存放目录，类型str
    :param name: 保存名称，类型str
    :return:
    """
    folder_address = osp.join(root, name)  # 文件夹路径

    # 保存的文件名
    name_x = osp.join(name + "_x.pt")
    name_edge_index = osp.join(name + "_edge_index.pt")
    name_y = osp.join(name + "_y.pt")
    name_train_mask = osp.join(name + "_train_mask.pt")
    name_test_mask = osp.join(name + "_test_mask.pt")

    # 保存文件的路径
    address_x = osp.join(folder_address, name_x)
    address_edge_index = osp.join(folder_address, name_edge_index)
    address_y = osp.join(folder_address, name_y)
    address_train_mask = osp.join(folder_address, name_train_mask)
    address_test_mask = osp.join(folder_address, name_test_mask)
    x = torch.load(address_x)
    edge_index = torch.load(address_edge_index)
    y = torch.load(address_y)
    train_mask = torch.load(address_train_mask)
    test_mask = torch.load(address_test_mask)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


def tran_tg_x_seq(x, x_length, y_length, limit=None):
    """
    将torch geometric格式的data的x变成rnn的训练格式
    :param x: torch geometric格式的data的x
    :param x_length: 数据的采集时间长度，类型int
    :param y_length: 标签的采集时间长度，类型int
    :param limit: 考虑采集时间的取值范围，类型list
    :return:
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if limit is not None:
        limit_one, limit_two = limit[0], limit[1]
        x = x[:, limit_one:limit_two]
    num_time = x.shape[1]  # 日期数量
    i, seq = 0, []
    while True:
        if (i + x_length + y_length) > num_time:
            break
        data_one = x[:, i: (i + x_length)]
        target_one = x[:, (i + x_length): (i + x_length + y_length)]
        seq.append((data_one, target_one))
        i = i + 1
    return seq


# 由邻接矩阵，计算度矩阵
def get_degree_matrix(adm, diag=False):
    num_nodes = adm.shape[0]  # 节点数量
    degree_matrix = []
    for i in range(num_nodes):
        node = adm[i, :]  # 当前节点
        nei = np.argwhere(node).reshape(-1)  # 当前节点的邻接点
        num_nei = nei.shape[0]  # 邻接点数量
        degree_matrix.append(num_nei)
    degree_matrix = np.array(degree_matrix)
    if diag:  # 取度矩阵为对角矩阵
        degree_matrix = np.diag(degree_matrix)
    return degree_matrix


# 由邻接矩阵和度矩阵，计算拉普拉斯矩阵（可选择是否归一化）
def get_laplacian_matrix(adm, dm, norm=False):
    if norm:  # 需要对拉普拉斯矩阵进行归一化
        dm_else = np.diag(np.power(dm, -0.5))  # 度矩阵的-0.5次方
        lam = np.diag(dm) - adm
        lam = np.dot(np.dot(dm_else, lam), dm_else)
    else:
        lam = np.diag(dm) - adm
    return lam


# 《Time-Varying_Graph_Signal_Reconstruction》(8)
def temporal_diff_operator(t):
    dh = np.zeros(shape=(t, t - 1))
    for i in range(t - 1):
        dh[i, i] = -1
        dh[i + 1, i] = 1
    return dh


def total_variation(adm, x, norm=False, time="one"):  # 计算全变差
    num_time = x.shape[1]  # 采集时刻的数量
    dm = get_degree_matrix(adm)  # 计算度矩阵
    lam = get_laplacian_matrix(adm, dm, norm)  # 拉普拉斯矩阵
    if time == "one":  # 对每个时刻的数据，计算全变差
        tv = []
        for i in range(num_time):
            x_one = x[:, i]
            x_one_row, x_one_col = x_one.reshape(1, -1), x_one.reshape(-1, 1)
            tv_one = float(np.dot(np.dot(x_one_row, lam), x_one_col))
            tv.append(tv_one)
        return tv
    elif time == "all":  # 对整个数据，计算全变差
        tv = np.trace(np.dot(np.dot(x.T, lam), x))
        return tv
    else:
        raise TypeError("time must be 'one' or 'all', but got {}".format(time))


# 由图结构的邻接矩阵，基于节点的坐标，计算节点之间的距离
def get_distance_of_adm(adm, xy):
    if adm.shape[0] != xy.shape[0]:
        raise AttributeError("adm.shape[0] muse equal to xy.shape[0]")
    dis_matrix = np.zeros(shape=adm.shape)  # 初始化距离矩阵
    dis_list = []  # 初始化距离列表
    u, v = np.nonzero(adm)
    for i in range(u.shape[0]):
        u_one, v_one = u[i], v[i]
        u_one_xy, v_one_xy = xy[u_one, :], xy[v_one, :]
        dist = np.linalg.norm(u_one_xy - v_one_xy)
        if dist == 0:  # 难以相信，竟然有坐标完全一致的节点
            dist = 0.0001  # 为了让它们的距离不为0，只能赋一个很小的值
        dis_matrix[u_one, v_one] = dist
        dis_list.append(dist)
    a, b = np.nonzero(dis_matrix)
    return dis_list, dis_matrix


# 使用图神经网络进行时间序列的回归预测
class GNNTime(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_weight, gnn_style, k_che=2, k_ap=10, alpha=0.5):
        super(GNNTime, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_weight = nn.Parameter(edge_weight)  # 初始化网络权重
        self.gnn_style = gnn_style  # 图神经网络类型
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, output_dim)
        self.che1 = gnn.ChebConv(input_dim, hidden_dim, K=k_che)
        self.che2 = gnn.ChebConv(hidden_dim, output_dim, K=k_che)
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, output_dim)
        self.graph1 = gnn.GraphConv(input_dim, hidden_dim)
        self.graph2 = gnn.GraphConv(hidden_dim, output_dim)
        self.gat1 = gnn.GATConv(input_dim, hidden_dim)
        self.gat2 = gnn.GATConv(hidden_dim, output_dim)
        self.gatv1 = gnn.GATv2Conv(input_dim, hidden_dim)
        self.gatv2 = gnn.GATv2Conv(hidden_dim, output_dim)
        self.tran1 = gnn.TransformerConv(input_dim, hidden_dim)
        self.tran2 = gnn.TransformerConv(hidden_dim, output_dim)
        self.agnn1 = gnn.AGNNConv(input_dim, hidden_dim)
        self.agnn2 = gnn.AGNNConv(hidden_dim, output_dim)
        self.tag1 = gnn.TAGConv(input_dim, hidden_dim)
        self.tag2 = gnn.TAGConv(hidden_dim, output_dim)
        self.arma1 = gnn.ARMAConv(input_dim, hidden_dim)
        self.arma2 = gnn.ARMAConv(hidden_dim, output_dim)
        self.cg1 = gnn.CGConv(input_dim, hidden_dim)
        self.cg2 = gnn.CGConv(hidden_dim, output_dim)
        self.appnp = gnn.APPNP(k_ap, alpha)
        self.grav1 = gnn.GravNetConv(input_dim, hidden_dim, 3, input_dim, 6)
        self.grav2 = gnn.GravNetConv(hidden_dim, output_dim, 3, input_dim, 6)
        self.gated1 = gnn.GatedGraphConv(input_dim, 10)
        self.gated2 = gnn.GatedGraphConv(output_dim, 10)
        self.res1 = gnn.ResGatedGraphConv(input_dim, hidden_dim)
        self.res2 = gnn.ResGatedGraphConv(hidden_dim, output_dim)
        self.gin = gnn.GINConv(nn.Linear(input_dim, output_dim))
        self.gine = gnn.GINEConv(nn.Linear(input_dim, output_dim))
        self.sg1 = gnn.SGConv(input_dim, hidden_dim)
        self.sg2 = gnn.SGConv(hidden_dim, output_dim)
        self.mf1 = gnn.MFConv(input_dim, hidden_dim)
        self.mf2 = gnn.MFConv(hidden_dim, output_dim)
        self.rgcn1 = gnn.RGCNConv(input_dim, hidden_dim, 10)
        self.rgcn2 = gnn.RGCNConv(hidden_dim, output_dim, 10)
        self.clu1 = gnn.ClusterGCNConv(input_dim, hidden_dim)
        self.clu2 = gnn.ClusterGCNConv(hidden_dim, output_dim)
        self.gcn_2 = gnn.GCN2Conv(input_dim, alpha=0.5)

    def forward(self, x, edge_index):
        if self.gnn_style == "gcn":
            h = self.gcn1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            # h = F.dropout(h, training=self.training)
            h = self.gcn2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "che":
            h = self.che1(x, edge_index)
            h = F.relu(h)
            h = self.che2(h, edge_index)
        elif self.gnn_style == "sage":
            h = self.sage1(x, edge_index)
            h = F.relu(h)
            h = self.sage2(h, edge_index)
        elif self.gnn_style == "graph":
            h = self.graph1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            h = self.graph2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "gat":  # 图注意力网络
            h = self.gat1(x, edge_index)
            h = F.relu(h)
            h = self.gat2(h, edge_index)
        elif self.gnn_style == "gatv":
            h = self.gatv1(x, edge_index)
            h = F.relu(h)
            h = self.gatv2(h, edge_index)
        elif self.gnn_style == "tran":
            h = self.tran1(x, edge_index)
            h = F.relu(h)
            h = self.tran2(h, edge_index)
        elif self.gnn_style == "agnn":
            h = self.agnn1(x, edge_index)
            h = F.relu(h)
            h = self.agnn2(h, edge_index)
        elif self.gnn_style == "tag":
            h = self.tag1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            h = self.tag2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "arma":
            h = self.arma1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            h = self.arma2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "cg":  # 需要工作站，笔记本跑不动
            h = self.cg1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            h = self.cg2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "appnp":  # 梯度居然中途消失了，无法反向传播
            h = self.appnp(x, edge_index)
            h = F.relu(h)
            h = self.appnp(h, edge_index)
        elif self.gnn_style == "grav":  # 运行比较慢
            h = self.grav1(x)
            h = F.relu(h)
            h = self.grav2(h)
        elif self.gnn_style == "gated":  # 无法运行，不知道哪里有问题
            h = self.gat1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            h = self.gat2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "res":
            h = self.res1(x, edge_index)
            h = F.relu(h)
            h = self.gat2(h, edge_index)
        elif self.gnn_style == "gin":
            h = self.gin(x, edge_index)
            h = F.relu(h)
            h = self.gin(h, edge_index)
        elif self.gnn_style == "gine":  # 无法运行
            h = self.gine(x, edge_index)
            h = F.relu(h)
            h = self.gine(h, edge_index)
        elif self.gnn_style == "sg":
            h = self.sg1(x, edge_index, self.edge_weight)
            h = F.relu(h)
            h = self.sg2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "mf":
            h = self.mf1(x, edge_index)
            h = F.relu(h)
            h = self.mf2(h, edge_index)
        elif self.gnn_style == "rgcn":  # 无法运行
            h = self.rgcn1(x, edge_index)
            h = F.relu(h)
            h = self.rgcn2(h, edge_index)
        elif self.gnn_style == "clu":
            h = self.clu1(x, edge_index)
            h = F.relu(h)
            h = self.clu2(h, edge_index)
        elif self.gnn_style == "gcn_2":  # 要求x_length(input_dim)与y_length(output_dim)一致，无法运行
            if self.input_dim != self.output_dim:
                raise ValueError("input_dim must be equal to output_dim")
            h = self.gcn_2(x, edge_index)
            h = F.relu(h)
            h = self.gcn_2(h, edge_index)
        else:
            raise TypeError("{} is unknown for gnn style".format(self.gnn_style))
        # h[-1] = 0  # 不清楚原因，输出最后一个值总是较大（增加此行不合逻辑，但结果图像看上去会好许多）
        return h


# 生成特征向量矩阵为DFT矩阵的算子，即传统时间序列有向图，对应的邻接矩阵
def time_series_graph(n):
    matrix = np.zeros(shape=(n, n))
    matrix[0, n - 1] = 1.
    for i in range(n - 1):
        matrix[i + 1, i] = 1.
    return matrix


def time_series_graph_k(n, k):
    """
    生成类似时间序列循环图的邻接矩阵，每一个节点，与其之后的若干节点相连
    :param n: 节点数量，类型int
    :param k: 与之后的邻接节点的数量，类型int
    :return:
    """
    adm = np.zeros(shape=(n, n))
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    else:
        for i in range(n):
            if i < (n - k):  # 直接向后连接就完事
                for k_one in range(1, k + 1):
                    adm[i, i + k_one] = 1.
            else:  # 将产生向最开始的节点连接的边
                for k_one in range(1, k + 1):
                    if (k_one + i) >= n:
                        mod = (k_one + i) % n  # 余数
                        adm[i, mod] = 1.
                    else:
                        adm[i, i + k_one] = 1.
    return adm


def get_adm_for_gnn_one_well(adm_style, n, k, time_series=None):
    """
    为gnn_one_well.py，选择图结构，获得其邻接矩阵
    :param adm_style: 图结构种类，类型str
    :param n: 节点数量，类型int
    :param k: 时间序列，各个节点前后邻接点的数量，类型int
    :param time_series: 时间序列，类型array，nvg图和hvg图需要的输入
    :return: 邻接矩阵，类型array
    """
    if adm_style == "dft":  # 时间序列循环图的移位算子矩阵，其特征向量矩阵为DFT矩阵
        adm = time_series_graph(n)
        k = 1
    elif adm_style == "ts":  # 时间序列循环图直观意义上的邻接矩阵，为上一行邻接矩阵的转置
        adm = time_series_graph_k(n, k)
    elif adm_style == "hvg":  # Horizontal Visibility Graph
        if time_series is None:
            raise ValueError("If adm_style is 'hvg', time_series can't be None")
        elif torch.is_tensor(time_series):  # 是tensor形式
            time_series = time_series.detach().cpu().numpy()
        adm, edge_index = get_hvg_adm(time_series.reshape(-1))
        k = np.nan
    elif adm_style == "nvg":  # Natural Visibility Graph
        if time_series is None:
            raise ValueError("If adm_style is 'nvg', time_series can't be None")
        elif torch.is_tensor(time_series):  # 是tensor形式
            time_series = time_series.detach().cpu().numpy()
        adm, edge_index = get_nvg_adm(time_series.reshape(-1))
        k = np.nan
    elif adm_style == "path":
        adm, edge_index = path_graph(time_series.reshape(-1))
        k = np.nan
    else:
        raise TypeError("adm_style must be 'dft' or 'ts', but got {}".format(adm_style))
    return adm, k


# 绘制一个数组，用像素点表示
def plot_mat_dynamic_data(data, x_name=None, y_name=None, title=None, fig_size=(12, 8)):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    im = ax.imshow(data)  # 绘制的图对象
    if x_name is None:
        ax.set_xticklabels(x_name, fontsize=20)  # 设置横轴名称
    if y_name is None:
        ax.set_yticklabels(y_name, fontsize=20)
    if title is None:
        ax.set_title(title, fontsize=20)
    fig.colorbar(im, ax=ax)  # 显示颜色柱
    return None


# 计算时间序列的HVG图，返回torch_geometric格式以及对应的邻接矩阵数组
def get_hvg_adm(time_series):
    m = time_series.shape[0]
    u, v, u_max, adm = [], [], 0, np.zeros(shape=(m, m))
    for i in range(m):
        if i == (m - 1):  # 最后一个节点
            continue
        for j in range(i + 1, m):
            if j == (i + 1):  # 相邻点必相连
                u.append(i), v.append(j), u.append(j), v.append(i)
                adm[i, j], adm[j, i] = 1., 1.
                u_max = time_series[j]
                if time_series[j] > time_series[i]:
                    break
            else:  # 不相邻节点
                if time_series[j] < u_max:
                    continue
                elif time_series[j] > u_max:
                    u.append(i), v.append(j), u.append(j), v.append(i)
                    adm[i, j], adm[j, i] = 1., 1.
                    if time_series[j] < time_series[i]:
                        u_max = time_series[j]
                    else:
                        break
    u = np.array(u).reshape(1, -1).astype(np.int64)
    v = np.array(v).reshape(1, -1).astype(np.int64)
    edge_index = np.vstack([u, v])
    edge_index = torch.from_numpy(edge_index)
    return adm, edge_index


# 尝试降低时间复杂度，计算时间序列的NVG图，返回torch_geometric格式以及对应的邻接矩阵数组
def get_nvg_adm(time_series):
    m = time_series.shape[0]
    u, v, u_max, adm = [], [], 0, np.zeros(shape=(m, m))
    for i in range(m):
        if i == (m - 1):  # 最后一个节点
            break
        for j in range(i + 1, m):
            if j == (i + 1):  # 相邻点必相连
                u.append(i), v.append(j), u.append(j), v.append(i)
                u_max = time_series[j]
                adm[i, j], adm[j, i] = 1., 1.
            else:  # 不相邻点
                if (time_series[j] - u_max) > (u_max - time_series[i]):
                    u.append(i), v.append(j), u.append(j), v.append(i)
                    u_max = time_series[j]
                    adm[i, j], adm[j, i] = 1., 1.
                else:
                    pass
    u = np.array(u).reshape(1, -1)
    v = np.array(v).reshape(1, -1)
    edge_index = np.vstack([u, v])
    edge_index = torch.from_numpy(edge_index)
    return adm, edge_index


# 将[2011 1 1]这种类型的array改变为'2011年1月1日'
def change_time_list(time_arr):
    year, month, day = time_arr[0], time_arr[1], time_arr[2]
    result = str(year) + "年" + str(month) + "月" + str(day) + "日"
    return result


# 对dataframe求平均
def pd_mean(df, features=None):
    if features is not None:  # 考虑部分特征
        df_selected = df.loc[:, features]
    else:  # 考虑全部特征
        df_selected = df
    col_name = df.columns.tolist()
    value = df_selected.values.astype(np.float)
    value_mean = np.mean(value, axis=0)
    value_mean = pd.DataFrame(value_mean.reshape(1, -1), columns=col_name)
    return value_mean


# 自定义损失函数，Frobenius范数
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output):
        loss = torch.linalg.norm(output)
        return loss


# 生成路径图
def path_graph(time_series):
    m = time_series.shape[0]
    u, v = [], []
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i, i + 1] = 1
        u.append(i), v.append(i + 1)
    u, v = np.array(u).astype(np.int64).reshape(1, -1), np.array(v).astype(np.int64).reshape(1, -1)
    edge_index = torch.from_numpy(np.vstack([u, v]))
    return adm, edge_index


def plot_clu_result(data, label, k, title=None, fig_size=(12, 8)):
    """
    绘制聚类结果，某个簇对应的曲线
    :param data: 用于聚类的数据，形式array
    :param label: 聚类后，各样本的所属簇类别，形式array
    :param k: 绘制的簇类别，类型int
    :param title: 绘图图像的标题，类型str
    :param fig_size: 绘制图像大小
    :return:
    """
    plt.figure(figsize=fig_size)
    for i in range(label.shape[0]):
        label_one = label[i]
        if label_one == k:
            plt.plot(data[i], alpha=0.5)
    if title is not None:
        plt.title(title, fontsize=15)


# 绘制所有的聚类结果
def plot_clu_result_all(data, label, title=None, fig_size=(12, 8)):
    num_label_class = pd.DataFrame(label).value_counts().shape[0]  # 聚类的簇的数量
    for i in range(num_label_class):
        title_i = title + "，第{}类聚类结果".format(i)
        plot_clu_result(data, label, i, title_i, fig_size)


# 自定义图结构
class SelfGraph(Graph):
    def __init__(self, xy, a, vertex_size=30, labels=None, gtype='self_graph', edge_index=False, **kwargs):
        xy = xy  # 节点坐标
        A = a  # 邻接矩阵
        vertex_size = vertex_size  # 节点大小
        if edge_index:  # 计算edge_index形式
            self.edge_index, self.edge_weight = tran_adm_to_edge_index(adm=a)
        self.labels = labels  # 节点标签
        self.adm = a

        # 设定绘图范围，比节点坐标的最大值（最小值）大（小）1；设定节点大小
        y_max = np.min(xy[:, 0]) - 1
        y_min = np.max(xy[:, 0]) + 1
        x_max = np.max(xy[:, 1]) - 1
        x_min = np.min(xy[:, 1]) + 1
        plotting = {"limits": np.array([y_min, y_max, x_min, x_max]),
                    "vertex_size": vertex_size}

        super(SelfGraph, self).__init__(W=A, coords=xy, gtype=gtype, plotting=plotting, **kwargs)


# 由节点坐标，通过k近邻生成图
class KnnGraph(SelfGraph):
    def __init__(self, xy, k=2, loop=False, style="adm", norm=False, arr=True, s=30, labels=None, undirected=True):
        self.style = style  # 邻接矩阵 or csr_matrix稀疏矩阵
        self.norm = norm  # 拉普拉斯矩阵是否归一化
        self.arr = arr  # 是否计算邻接矩阵、拉普拉斯矩阵的array格式
        num_nodes = xy.shape[0]  # 节点数量

        # 通过节点坐标，获得k近邻图的邻接矩阵
        if isinstance(xy, np.ndarray):  # 若输入为数组
            xy_tensor = torch.from_numpy(xy)
        elif torch.is_tensor(xy):  # 若输入为张量
            xy_tensor = xy
            xy = xy.detach().cpu().numpy()
        batch = torch.tensor([0] * num_nodes)
        edge_index = knn_graph(xy_tensor, k=k, batch=batch, loop=loop)
        if undirected:  # 变成无向图的形式
            edge_index = tran_edge_index_to_undirected(edge_index)
        self.edge_index = edge_index

        if style == "csr":  # 邻接矩阵为稀疏矩阵形式（当节点数量过多时，需要花费相当多时间）
            edge_index = edge_index.detach().numpy()
            row = []
            for i in range(num_nodes):
                row_one = [i] * num_nodes
                row = row + row_one  # 行指标
            col = np.arange(num_nodes).tolist() * num_nodes  # 列指标
            row_col = np.hstack([np.array(row).reshape(-1, 1), np.array(col).reshape(-1, 1)])

            num_edges = edge_index.shape[1]  # 边数量
            data = np.array([0] * num_nodes ** 2)
            j_all = []
            for i in range(num_edges):
                edge = edge_index[:, i]
                for j in range(num_edges ** 2):
                    row_col_one = row_col[j, :]
                    if (row_col_one == edge).all():
                        j_all.append(j)
                        break
            data[j_all] = 1
            matrix = csr_matrix((data.tolist(), (row, col)), shape=(num_nodes, num_nodes))  # 稀疏矩阵形式
        elif style == "adm":
            matrix = tran_edge_index_to_adm(edge_index=edge_index)
            if self.arr:
                self.adm = matrix
                self.lam = get_laplacian_matrix(adm=self.adm, norm=norm)
        else:
            raise TypeError("style must be 'adm' or 'csr', but got {}".format(style))

        gtype = "knn_graph"
        super(KnnGraph, self).__init__(xy=xy, a=matrix, gtype=gtype, vertex_size=s, labels=labels)


# 让edge_index变成无向图的形式
def tran_edge_index_to_undirected(edge_index):
    edge_index = edge_index.detach().cpu().numpy()
    num_edges = edge_index.shape[1]  # 此时，边的数量
    u, v = edge_index[0, :], edge_index[1, :]
    edge_all = []
    for i in range(num_edges):
        u_one, v_one = u[i], v[i]
        edge_one = (u_one, v_one)
        edge_all.append(edge_one)
        edge_one_pair = (v_one, u_one)
        edge_all.append(edge_one_pair)
        edge_all = list(set(edge_all))

    num_edges_true = len(edge_all)  # 边的真实数量
    u_all, v_all = [], []
    for i in range(num_edges_true):
        edge_one = edge_all[i]
        u_one, v_one = edge_one[0], edge_one[1]
        u_all.append(u_one)
        v_all.append(v_one)
    u_all, v_all = np.array(u_all).reshape(1, -1), np.array(v_all).reshape(1, -1)
    edge_index_true = np.vstack([u_all, v_all])
    edge_index_true = torch.from_numpy(edge_index_true)

    return edge_index_true


# 若两节点的欧氏距离小于sigma，则连接此两节点
class MaxDisGraph(SelfGraph):
    def __init__(self, xy, name, root="static_data/new_SD", sigma=1, arr=True, s=30, labels=None, norm=False,
                 undirected=True):
        self.sigma = sigma  # 设定的最大距离
        dis_matrix_address = osp.join(root, name + "_dis_matrix.pkl")  # 获得坐标距离矩阵
        if osp.exists(dis_matrix_address):  # 若之前已计算，直接读取坐标距离矩阵
            with open(dis_matrix_address, "rb") as f:
                dis_matrix = pickle.load(f)
        else:  # 若之前未计算，则计算坐标距离矩阵并保存
            dis_matrix = get_xy_distance(xy)
            with open(dis_matrix_address, "wb") as f:
                pickle.dump(dis_matrix, f)

        index = np.argwhere(dis_matrix <= sigma)  # 找出距离小于sigma的所有边
        i_all, j_all = [], []
        for k in range(index.shape[0]):
            edge_index_one = index[k, :]
            i, j = edge_index_one[0], edge_index_one[1]
            if i == j:
                continue
            else:
                i_all.append(i)
                j_all.append(j)
        i_all, j_all = np.array(i_all).reshape(1, -1), np.array(j_all).reshape(1, -1)
        edge_index = np.vstack([i_all, j_all])
        edge_index = torch.from_numpy(edge_index)
        if undirected:  # 变成无向图的形式
            edge_index = tran_edge_index_to_undirected(edge_index)
        self.edge_index = edge_index
        adm = tran_edge_index_to_adm(edge_index, num_nodes_truth=xy.shape[0])
        if arr:
            self.adm = adm
            self.lam = get_laplacian_matrix(adm=self.adm, norm=norm)

        gtype = "MaxDisGraph"
        super(MaxDisGraph, self).__init__(xy=xy, a=adm, gtype=gtype, vertex_size=s, labels=labels)


def get_xy_distance(xy):  # 计算节点之间的距离
    num_nodes = xy.shape[0]  # 节点数量
    dis_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i in range(num_nodes):
        xy_i = xy[i, :]
        for j in range(i + 1, num_nodes):
            xy_j = xy[j, :]
            distance = np.linalg.norm(xy_j - xy_i)
            dis_matrix[i, j], dis_matrix[j, i] = distance, distance
    return dis_matrix


# 德洛内三角划分得到的图结构
class DLNGraph(SelfGraph):
    def __init__(self, xy, norm=False, arr=True, s=30, labels=None):
        self.arr = arr  # 是否计算邻接矩阵、拉普拉斯矩阵的array格式
        num_nodes = xy.shape[0]  # 节点数量
        self.edge_index = delaunay_edge_index(xy=xy)  # 边索引
        adm = tran_edge_index_to_adm(self.edge_index, num_nodes).astype(np.int64)

        if self.arr:  # 添加计算邻接矩阵、拉普拉斯矩阵的array格式
            self.adm = adm
            self.lam = get_laplacian_matrix(adm=self.adm, norm=norm)
        gtype = "Delaunay_Graph"
        super(DLNGraph, self).__init__(xy=xy, a=adm, gtype=gtype, vertex_size=s, labels=labels)


# 提出出，经过德洛内三角剖分后得到的三角形，的边索引
def delaunay_edge_index(xy):
    dln = Delaunay(xy)
    tri = dln.simplices  # 三角形结构
    num_tri = tri.shape[0]  # 三角形的数量
    edge_index_list = []
    for i in range(num_tri):  # 获得三角形的所有边
        tri_one = tri[i, :]
        edge_index_list_one = [(tri_one[0], tri_one[1]), (tri_one[1], tri_one[0]), (tri_one[0], tri_one[2]),
                               (tri_one[2], tri_one[0]), (tri_one[1], tri_one[2]), (tri_one[2], tri_one[1])]
        edge_index_list = edge_index_list + edge_index_list_one
        edge_index_list = list(set(edge_index_list))  # 消除重复的边索引
    num_edges = len(edge_index_list)
    u, v = [], []
    for i in range(num_edges):
        edge_index_one = edge_index_list[i]
        u_one, v_one = edge_index_one[0], edge_index_one[1]
        u.append(u_one)
        v.append(v_one)
    u = np.array(u).astype(np.int64).reshape(1, -1)
    v = np.array(v).astype(np.int64).reshape(1, -1)
    edge_index = torch.from_numpy(np.vstack([u, v]))
    return edge_index
