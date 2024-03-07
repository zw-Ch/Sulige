import os
import pandas as pd
import os.path as osp
import numpy as np
import xlrd


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
