import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('..')
import func.utils as uti
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def clean_data(data, thre):
    data_new = []
    for i in range(len(data)):
        if data[i] < thre:
            if (i == 0) or (i == (len(data) - 1)):
                data_new.append(data[i])
            else:
                val = (data[i - 1] + data[i + 1]) / 2
                data_new.append(val)
        else:
            data_new.append(data[i])
    data_new = np.array(data_new)
    return data_new


data_ad = '../dataset'
name = '苏177'

df = uti.get_sheet(osp.join(data_ad, 'product_data'), name)
op = df.loc[:, '油压(MPa)'].values.reshape(-1)
cp = df.loc[:, '套压(MPa)'].values.reshape(-1)
dgo = df.loc[:, '日产气量(10⁴m³/d)'].values.reshape(-1)
date = df.loc[:, '日期'].values.reshape(-1)
to = 2000
thre = 1

"""
绘制归一化前后的油压与日产气量
"""
# scaler = MinMaxScaler()
# op_norm = scaler.fit_transform(op.reshape(-1, 1)).reshape(-1)
# dgo_norm = scaler.fit_transform(dgo.reshape(-1, 1)).reshape(-1)

# _, ax1 = plt.subplots(figsize=(18, 12))
# plt.sca(ax1)
# p1 = plt.plot(op[:to], label="油压", c='red')
# p2 = plt.plot(op_norm[:to] + 0.5, label="归一化后油压", c='yellow')
# plt.xlabel("日期", fontsize=25)
# plt.xticks(fontsize=25)
# plt.ylabel("油压，MPa", fontsize=25)
# plt.yticks(fontsize=25)
#
# plt.sca(ax1.twinx())
# p3 = plt.plot(dgo[:to], label="日产气量", c='blue')
# p4 = plt.plot(dgo_norm[:to], label="归一化后日产气量", c='purple')
# # plt.title(title, fontsize=20)
# plt.ylabel("日产气量，$10^4m^3/$天", fontsize=25)
# plt.yticks(fontsize=25)
#
# p = p1 + p2 + p3 + p4
# labs = [p_one.get_label() for p_one in p]
# ax1.legend(p, labs, loc=0, fontsize=25)

"""
绘制数据清洗前后的油压and日产气量
"""
op_new, dgo_new = clean_data(op, thre), clean_data(dgo, thre)

_, ax1 = plt.subplots(figsize=(18, 12))
plt.sca(ax1)
p1 = plt.plot(op[:to], label="油压", c='red')
plt.xlabel("日期", fontsize=25)
plt.xticks(fontsize=25)
plt.ylabel("油压，MPa", fontsize=25)
plt.yticks(fontsize=25)

plt.sca(ax1.twinx())
p2 = plt.plot(dgo[:to], label="日产气量", c='blue')
plt.ylabel("日产气量，$10^4m^3/$天", fontsize=25)
plt.yticks(fontsize=25)

plt.title("原始监测数据", fontsize=25)
p = p1 + p2
labs = [p_one.get_label() for p_one in p]
ax1.legend(p, labs, loc=0, fontsize=25)


_, ax1 = plt.subplots(figsize=(18, 12))
plt.sca(ax1)
p1 = plt.plot(op_new[:to], label="油压", c='red')
plt.xlabel("日期", fontsize=25)
plt.xticks(fontsize=25)
plt.ylabel("油压，MPa", fontsize=25)
plt.yticks(fontsize=25)

plt.sca(ax1.twinx())
p2 = plt.plot(dgo_new[:to], label="日产气量", c='blue')
plt.ylabel("日产气量，$10^4m^3/$天", fontsize=25)
plt.yticks(fontsize=25)

plt.title("清洗后数据", fontsize=25)
p = p1 + p2
labs = [p_one.get_label() for p_one in p]
ax1.legend(p, labs, loc=0, fontsize=25)


print()
plt.show()
print()
