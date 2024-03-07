"""
从原始的xls文件中读取日产气量
厚度、电阻率、声波时差、岩石密度、泥质含量、总孔隙度、渗透率、含油气饱和度
"""
import pickle
import numpy as np
import pandas as pd
import xlrd
import os
import os.path as osp
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import func.process as pro


xls_ad = '../dataset/product_data'
name_list = os.listdir(xls_ad)
num_xls = len(name_list)

names, dates, pts, cps, ops, dgos = [], [], [], [], [], []
for i in range(num_xls):
    xls_name = name_list[i]
    print(osp.join(xls_ad, xls_name))
    xls = xlrd.open_workbook(osp.join(xls_ad, xls_name), on_demand=True)
    sheet_name_list = xls.sheet_names()
    num_sheet = len(sheet_name_list)

    print("{}{}{}".format('-' * 30, xls_name, '-' * 30))
    for j in range(num_sheet):
        sheet_name = sheet_name_list[j]
        if sheet_name == '井名列表':
            continue
        df = pd.read_excel(osp.join(xls_ad, xls_name), sheet_name=sheet_name)
        name = df.loc[:, '井号'].values[0]                 # 油井名称
        date = df.loc[:, '日期'].values.reshape(-1)        # 日期
        pt = df.loc[:, '生产时间(h)'].values.reshape(-1)      # 生产时间
        cp = df.loc[:, '套压(MPa)'].values.reshape(-1)
        op = df.loc[:, '油压(MPa)'].values.reshape(-1)
        dgo = df.loc[:, '日产气量(10⁴m³/d)'].values.reshape(-1)

        names.append(name), dates.append(date), pts.append(pt)
        cps.append(cp), ops.append(op), dgos.append(dgo)
        print('{}，读取完毕'.format(name))

product = pro.Product(names, dates, pts, cps, ops, dgos)
with open('../dataset/product_all.pkl', 'wb') as f:
    pickle.dump(product, f)

print()
plt.show()
print()
