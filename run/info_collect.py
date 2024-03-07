"""
统计信息
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import os
import xlrd
import pickle
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
        return None
    return df


def get_wells(root, bi_class, dyc_class, block_class):
    dyc = pd.read_csv(osp.join(root, '动态分类1.csv'))
    bi_ = dyc.loc[:, "井型"].values.reshape(-1)
    block_ = dyc.loc[:, "区块"].values.reshape(-1)
    dyc_ = dyc.loc[:, "分类"].values.reshape(-1)
    wells = dyc.loc[:, "井号"].values.reshape(-1)
    if block_class == "":
        idx = np.argwhere((bi_ == bi_class) & (dyc_ == dyc_class)).reshape(-1)
    elif type(block_class) == list:
        if len(block_class) == 2:
            idx = np.argwhere((bi_ == bi_class) & (dyc_ == dyc_class) & ((block_ == block_class[0]) | (block_ == block_class[1]))).reshape(-1)
    else:
        idx = np.argwhere((bi_ == bi_class) & (dyc_ == dyc_class) & (block_ == block_class)).reshape(
            -1)
    wells = wells[idx]
    return wells


def get_info(cps, dgos):
    m = len(cps)
    print("{}\n{}\n{}\n数量：{}".format(block_class, bi_class, dyc_class, m))

    # 初期
    idx_start = 330
    cp, cp_diff, dgo_mean = 0, 0, 0
    for i in range(m):
        cp_, dgo_ = cps[i], dgos[i]
        cp = cp + cp_[idx_start]
        cp_diff = cp_diff + np.abs(cp_[idx_start] - cp_[idx_start - 1])
        dgo_mean = dgo_mean + np.mean(dgo_)
    cp = cp / m
    cp_diff = cp_diff / m
    dgo_mean = dgo_mean / m
    print("初期\t套压：{:.2f}\t压降速率：{:.3f}\t平均日产：{:.2f}".format(cp, cp_diff, dgo_mean))

    # 990天
    idx_mid = 990
    cp, cp_diff, dgo_mean, dgo_diff, dgo_sum, dgo_sum_all = 0, 0, 0, 0, 0, 0
    for i in range(m):
        cp_, dgo_ = cps[i], dgos[i]
        cp = cp + cp_[idx_mid]
        cp_diff = cp_diff + np.abs(cp_[idx_mid] - cp_[idx_mid - 1])
        dgo_mean = dgo_mean + np.mean(dgo_[:idx_mid])
        dgo_diff = dgo_diff + (np.sum(dgo_[:365]) - np.sum(dgo_[365:730])) / np.sum(dgo_[:365]) * 100
        dgo_sum = dgo_sum + np.sum(dgo_[:idx_mid])
        dgo_sum_all = dgo_sum_all + np.sum(dgo_)
    cp = cp / m
    cp_diff = cp_diff / m
    dgo_mean = dgo_mean / m
    dgo_diff = dgo_diff / m
    dgo_sum = dgo_sum / m
    dgo_sum_all = dgo_sum_all / m
    print("中期\t套压：{:.2f}\t压降速率：{:.3f}\t平均日产：{:.2f}\t平均年递减率：{:.2f}\t累积产量：{:.0f}\t预测累产：{:.0f}".
          format(cp, cp_diff, dgo_mean, dgo_diff, dgo_sum, dgo_sum_all))


fig_si = (16, 16)
fo_si = 35
fo_bar_si = 20
fo_ti_si = 15

root = "../dataset"
bi_class_list = ["直井", "水平井"]
bi_class = bi_class_list[1]                # 可选："直井", "水平井"
dyc_class_list = ["Ⅰ类井", "Ⅱ类井", "Ⅲ类井"]
dyc_class = dyc_class_list[1]             # 可选："Ⅰ类井", "Ⅱ类井", "Ⅲ类井"

"""
'苏14区', '苏47区', '桃7区', '苏46区', '苏59区', '苏75区', '苏11区', 
'苏48区', '苏20区', '桃2区', '苏120区', '苏19区'
"""
block_class = ["苏120区", "苏47区"]

"""
读取所有文件数据
"""
with open(osp.join(root, 'product_all.pkl'), 'rb') as f:
    product_all = pickle.load(f)
ops_ori = product_all.ops                           # 所有油压
dgos_ori = product_all.dgos                         # 所有油井的日产气量
cps_ori = product_all.cps
names_ori = np.array(product_all.names)             # 所有油井的名称
dates_ori = product_all.dates                       # 所有油井的生产日期

wells = get_wells(osp.join(root, "class_info"), bi_class, dyc_class, block_class)
idx = np.isin(names_ori, wells)

ops, dgos, cps, names, dates = [], [], [], [], []
for i in range(idx.shape[0]):
    idx_i = idx[i]
    if idx_i:
        op, dgo, cp, name, date = ops_ori[i], dgos_ori[i], cps_ori[i], names_ori[i], dates_ori[i]
        idx_rz = np.argwhere(dgo != 0).reshape(-1)
        op, dgo, cp, date = op[idx_rz], dgo[idx_rz], cp[idx_rz], date[idx_rz]
        if cp.shape[0] <= 990:
            continue
        ops.append(op), dgos.append(dgo), cps.append(cp), names.append(name), dates.append(date)

get_info(cps, dgos)

print()
plt.show()
print()
