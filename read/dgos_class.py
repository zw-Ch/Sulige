"""
措施连续、自然连续，分类
"""
import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np
import os
import os.path as osp
import sys
import pandas as pd
sys.path.append('..')
import func.process as pro


def tran_datetime_one(dt):
    ts = pd.to_datetime(str(dt))
    dt_str = ts.strftime('%Y.%m.%d')
    dt_str = dt_str.replace('.', '/')
    return dt_str


data_ad = '../dataset'

with open(osp.join(data_ad, 'product_all.pkl'), 'rb') as f:
    product_all = pickle.load(f)

names = product_all.names
dgos = product_all.dgos
dates = product_all.dates

info = pd.read_excel(osp.join(data_ad, 'class_info', '目前气井分类.xlsx'), sheet_name='分类')
info = info[info['分类'] == '自然连续']
names_info = info.loc[:, '井号'].values
times_info = info.loc[:, '投产日期'].values
# times_info = pro.tran_datetime(info.loc[:, '投产日期'].values)
num_info = names_info.shape[0]

names_, dgos_, times_ = [], [], []
for i in range(num_info):
    name, dgo, date = names[i], dgos[i], dates[i]
    idx = np.argwhere(names_info == name)
    if idx.shape[0] == 0:
        continue
    else:
        idx = idx.reshape(-1)[0]
        a = tran_datetime_one(times_info[idx])
        names_.append(name), dgos_.append(dgo)

plt.figure()
plt.plot(dgos_[1])

print()
plt.show()
print()
