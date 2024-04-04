import numpy as np
import pandas as pd
import torch
import xlrd
import os
import os.path as osp
import pickle
from torch.utils.data import Dataset, DataLoader


class Method:
    """
    Public methods of wells (List of Well Class) and wells_name (List of str)
    """
    def __init__(self):
        self.wells = []
        self.wells_name = []

    def get_date(self):
        date = [well.date for well in self.wells]
        return date

    def get_length(self):
        length = np.array([well.data.size for well in self.wells])
        return length

    def get_loc(self):
        loc = np.array([(well.x, well.y) for well in self.wells])
        return loc if len(loc) != 0 else np.zeros((0, 2))

    def get_data(self):
        data = [well.data for well in self.wells]
        return data

    def remain_idx(self, idx):
        if len(idx) == 0:
            return [], []
        else:
            pairs = [(self.wells[i], self.wells_name[i]) for i in idx]
            wells, wells_name = zip(*pairs)
            return list(wells), list(wells_name)

    def remain_loc(self, x_ran, y_ran):
        loc = self.get_loc()
        xs, ys = loc[:, 0], loc[:, 1]
        x_min, x_max, y_min, y_max = x_ran[0], x_ran[1], y_ran[0], y_ran[1]
        idx = np.argwhere((xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)).reshape(-1).tolist()
        if len(idx) == 0:
            raise ValueError(f"The (x, y) Range is [({np.min(xs)}, {np.min(ys)}), ({np.max(xs), np.max(ys)})], but got [({x_min}, {y_min}), ({x_max, y_max})]!")
        self.wells, self.wells_name = self.remain_idx(idx)
        return None

    def remain_length(self, length_ran):
        length = self.get_length()
        length_min, length_max = length_ran[0], length_ran[1]
        idx = np.argwhere((length >= length_min) & (length <= length_max)).reshape(-1).tolist()
        if len(idx) == 0:
            raise ValueError(f"The Length Range is [{np.min(length)}, {np.max(length)}], but got [{length_min}, {length_max}]!")
        self.wells, self.wells_name = self.remain_idx(idx)
        return None

    def remain_wells_name(self, wells_name):
        idx = [i for i, item in enumerate(self.wells_name) if item in wells_name]
        if len(idx) == 0:
            raise NameError("Not exist!")
        self.wells, self.wells_name = self.remain_idx(idx)
        return 0

    def remove_data_zero(self, value):
        """
        Remove zero value in data, and clean op, cp, date of each well

        :param value: float
        :return:
        """
        for well in self.wells:
            well.clean_data(value)
        return None

    def get_well_by_name(self, well_name):
        try:
            idx = self.wells_name.index(well_name)
        except ValueError:
            raise ValueError("'well_name' is not found!")
        return self.wells[idx]


class Well:
    """
    Information of single well
    """
    def __init__(self, well_name, date, cp, op, data, x, y):
        self.well_name = well_name
        self.date = date
        self.cp = cp
        self.op = op
        self.data = data
        self.x = x
        self.y = y

    def clean_data(self, value):
        """
        Remove record when data is smaller than given 'value'

        :param value: (float)
        :return:
        """
        idx = np.where(self.data <= value)[0]
        self.data = np.delete(self.data, idx)
        self.date = np.delete(self.date, idx)
        self.cp = np.delete(self.cp, idx)
        self.op = np.delete(self.op, idx)
        return None


class Wells(Method):
    """
    Information of multiple wells
    """
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.wells, self.wells_name = self.get_all_wells()

    def __str__(self):
        return f"Well Num: {len(self.wells)}"

    def get_all_wells(self):
        """
        Get Information of all wells

        :return:
        """
        wells, wells_name = [], []
        block_names = os.listdir(osp.join(self.root, "product_data"))
        block_names = [block_name.split('.')[0] for block_name in block_names]
        for block_name in block_names:
            block = Block(self.root, block_name)
            wells += block.wells
            wells_name += block.wells_name
        return wells, wells_name


class Block(Method):
    """
    Information of single block
    """
    def __init__(self, root, block_name):
        super().__init__()
        self.root = root
        self.block_name = block_name
        self.block_path = osp.join(self.root, "product_data", self.block_name + '.xls')
        self.wells, self.wells_name = self.get_wells_info()

    def __str__(self):
        return f"Block: {self.block_name}, Well Num: {len(self.wells)}"

    def get_well_info(self, well_name, x, y):
        """
        Get Information including 'well_name', 'x', and 'y' in one well

        :return: Well class
        """
        df = pd.read_excel(self.block_path, sheet_name=well_name)
        date = df['日期'].values.reshape(-1)
        cp = df['套压(MPa)'].values.reshape(-1)
        op = df['油压(MPa)'].values.reshape(-1)
        data = df['日产气量(10⁴m³/d)'].values.reshape(-1)
        well = Well(well_name, date, cp, op, data, x, y)
        # print("区块：{}，    油井：{}，    完成".format(self.block_name, well_name))
        return well

    def get_wells_info(self):
        """
        Get Information of all wells in this Block

        :return: List of Well Class
        """
        wells_name_ = xlrd.open_workbook(self.block_path, on_demand=True).sheet_names()
        info_path = osp.join(self.root, "info_{}.pkl".format(self.block_name))
        if osp.exists(info_path):
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
                return info['wells'], info['wells_name']
        pos_path = osp.join(self.root, "class_info", "well_class.csv")
        df = pd.read_csv(pos_path, index_col=0)
        df = df.dropna(subset=['x', 'y'])
        well_names_class = df.index.to_numpy()
        xs = df['x'].values.reshape(-1)
        ys = df['y'].values.reshape(-1)
        wells, wells_name = [], []
        for i, well_name in enumerate(wells_name_):
            idx = np.argwhere(well_names_class == well_name).reshape(-1)
            if idx.shape[0] == 1:
                well = self.get_well_info(well_name, xs[idx[0]], ys[idx[0]])
                wells.append(well), wells_name.append(well_name)
        with open(info_path, 'wb') as f:
            pickle.dump({'wells': wells, 'wells_name': wells_name}, f)
        return wells, wells_name


class Blocks(Method):
    """
    Information of multiple blocks
    """
    def __init__(self, root, blocks_name=None):
        super().__init__()
        self.root = root
        self.blocks_name = blocks_name
        self.wells, self.wells_name = self.get_all_blocks()

    def __str__(self):
        return f"Block Num: {len(self.blocks_name)}, Well Num: {len(self.wells)}"

    def get_all_blocks(self):
        """
        Get Wells Information from given 'blocks_name'

        :return:
        """
        if self.blocks_name is None:
            blocks_name = os.listdir(osp.join(self.root, "product_data"))
            blocks_name = [block_name.split('.')[0] for block_name in blocks_name]
            blocks_name.sort()
            self.blocks_name = blocks_name

        wells, wells_name = [], []
        for block_name in self.blocks_name:
            block = Block(self.root, block_name)
            print(block)
            wells += block.wells
            wells_name += block.wells_name
        return wells, wells_name


class SelfData(Dataset):
    def __init__(self, x, y, *args):
        super().__init__()
        self.x = x
        self.y = y
        self.args = args
        self.data_else = self.get_data_else()

    def __len__(self):
        return len(self.x)

    def get_data_else(self):
        num = len(self.args)
        data_else = [0] * num
        if num != 0:
            for i in range(num):
                data_else_ = self.args[i]
                data_else[i] = data_else_
        return data_else

    def __getitem__(self, idx):
        res = [get_idx_by_dim(self.x, idx), get_idx_by_dim(self.y, idx)]
        if len(self.data_else) != 0:
            num = len(self.data_else)
            data_else = [0] * num
            for i in range(num):
                x = self.data_else[i]
                data_else[i] = get_idx_by_dim(x, idx)
            res = res + data_else
        return tuple(res)


def get_idx_by_dim(data, idx):
    """
    Get element of single-index based on Input Dimension

    :param data: Input Data (List or torch.Tensor or numpy.ndarray)
    :param idx: single-index (int)
    :return:
    """
    if isinstance(data, list):
        return data[idx]
    elif torch.is_tensor(data):
        n_dim = data.dim()
    elif isinstance(data, np.ndarray):
        n_dim = data.ndim
    else:
        raise TypeError("The input must be torch.tensor or numpy.ndarray!")

    def recursive(data_, idx_, dim):
        if dim == 1:
            return data_[idx_]
        else:
            return recursive(data_[idx_, ...], Ellipsis, dim - 1)

    return recursive(data, idx, n_dim)


def get_xy(seqs, lx, ly):
    """
    Get x and y from time series

    :param seqs: List of numpy.ndarray
    :param lx: Length of x
    :param ly: Length of y
    :return:
    """
    xs, ys = [], []
    for seq in seqs:
        size = seq.shape[0] - lx - ly + 1
        x, y = torch.zeros(size, lx), torch.zeros(size, ly)
        for i in range(size):
            x[i, :] = torch.from_numpy(seq[i: i + lx]).float()
            y[i, :] = torch.from_numpy(seq[i + lx: i + lx + ly]).float()
        xs.append(x), ys.append(y)
    return xs, ys


def train_test(train_ratio, info):
    """
    Get Train and Test Sets

    :param train_ratio:
    :param info:
    :return:
    """
    num = len(info)
    train_num = int(num * train_ratio)
    return info[:train_num], info[train_num:]


def get_loader(data, wells_name, train_ratio, lx, ly):
    """
    Get Train and Test DataLoader

    :param data:
    :param wells_name:
    :param train_ratio:
    :param lx:
    :param ly:
    :return: torch.utils.data.DataLoader
    """
    train_data, test_data = train_test(train_ratio, data)
    train_name, test_name = train_test(train_ratio, wells_name)
    train_x, train_y = get_xy(train_data, lx, ly)
    test_x, test_y = get_xy(test_data, lx, ly)
    train_dataset = SelfData(train_x, train_y, train_name)
    test_dataset = SelfData(test_x, test_y, test_name)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)
    print(f"Train Number: {len(train_dataset)}")
    print(f"Test Number: {len(test_dataset)}")
    return train_loader, test_loader


def cal_rmse_1d(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def cal_rmse(true, pred):
    """
    Calculate Root Mean Square Error

    :param true:
    :param pred:
    :return:
    """
    rmse = [cal_rmse_1d(true_, pred_) for true_, pred_ in zip(true, pred)]
    return np.mean(rmse)


def cal_r2_1d(true, pred):
    corr_matrix = np.corrcoef(true, pred)
    corr = corr_matrix[0, 1]
    r2 = corr ** 2
    return r2


def cal_r2(true, pred):
    """
    Calculate Coefficient of Determination

    :param true:
    :param pred:
    :return:
    """
    r2 = [cal_r2_1d(true_, pred_) for true_, pred_ in zip(true, pred)]
    return np.mean(r2)
