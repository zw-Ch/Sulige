import numpy as np
import pandas as pd
import xlrd
import os
import os.path as osp
import pickle


class Well:
    def __init__(self, well_name, date, cp, op, data, x, y):
        self.well_name = well_name
        self.date = date
        self.cp = cp
        self.op = op
        self.data = data
        self.x = x
        self.y = y


class Block:
    def __init__(self, root, block_name):
        self.root = root
        self.block_name = block_name
        self.block_path = osp.join(self.root, "product_data", self.block_name + '.xls')
        self.wells_name = self.get_wells_name()
        self.wells = self.get_wells_info()

    def get_wells_name(self):
        """
        Get the name of all wells

        :return: numpy.ndarray
        """
        xls = xlrd.open_workbook(self.block_path, on_demand=True)
        return np.array(xls.sheet_names())

    def get_well_info(self, well_name, x, y):
        """
        Generate Well object by 'well_name', 'x', and 'y'

        :return: Well class
        """
        df = pd.read_excel(self.block_path, sheet_name=well_name)
        date = df['日期'].values.reshape(-1)
        cp = df['套压(MPa)'].values.reshape(-1)
        op = df['油压(MPa)'].values.reshape(-1)
        data = df['日产气量(10⁴m³/d)'].values.reshape(-1)
        well = Well(well_name, date, cp, op, data, x, y)
        print("区块：{}，    油井：{}，    完成".format(self.block_name, well_name))
        return well

    def get_wells_info(self):
        """
        Get Well objects in this Block

        :return:
        """
        info_path = osp.join(self.root, "info_{}.pkl".format(self.block_name))
        if osp.exists(info_path):
            with open(info_path, 'rb') as f:
                return pickle.load(f)
        pos_path = osp.join(self.root, "class_info", "well_class.csv")
        df = pd.read_csv(pos_path, index_col=0)
        df = df.dropna(subset=['x', 'y'])
        well_names_class = df.index.to_numpy()
        xs = df['x'].values.reshape(-1)
        ys = df['y'].values.reshape(-1)
        wells = []
        for i, well_name in enumerate(self.wells_name):
            idx = np.argwhere(well_names_class == well_name).reshape(-1)
            if idx.shape[0] == 1:
                well = self.get_well_info(well_name, xs[idx[0]], ys[idx[0]])
                wells.append(well)
        with open(info_path, 'wb') as f:
            pickle.dump(wells, f)
        return wells
