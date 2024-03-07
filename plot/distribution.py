"""
绘制概率密度
"""
import pickle
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
import sys
sys.path.append('..')
import func.process as pro


def get_max(x, y):
    idx = np.argmax(y)
    max_x, max_y = x[idx], y[idx]
    return max_x, max_y


def cal_2d_deri(x, y):
    dx, dy = np.diff(x), np.diff(y)
    d = dy / dx
    ddy = np.diff(d)
    dd = ddy / dx[1:]
    ip = []
    for i in range(1, dd.shape[0]):
        dd_last, dd_now = dd[i - 1], dd[i]
        if dd_last * dd_now < 0:
            ip.append(i)
    x_ip, y_ip = x[ip], y[ip]
    return x_ip[1], y_ip[1]


def cal_dist(x, bins):
    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_label = pd.DataFrame(x_label)
    x_label_vc = pd.DataFrame(x_label).value_counts()
    interval = x_label_vc.index.tolist()
    interval_sum = x_label_vc.values
    mid_all, left, right = [], float('inf'), -float('inf')
    for i in range(bins):
        interval_one = interval[i][0]
        left_one, right_one = interval_one.left, interval_one.right
        mid = (left_one + right_one) / 2
        mid_all.append(mid)
        if left_one < left:
            left = left_one
        if right_one > right:
            right = right_one
    mid_all = np.array(mid_all)
    sort_index = np.argsort(mid_all)
    mid_all_sort = mid_all[sort_index]
    mid_all_sort = np.around(mid_all_sort, 2)
    interval_sum_sort = interval_sum[(sort_index)]
    return mid_all_sort, interval_sum_sort, left, right


def plot_dist(x, bins, jump, pos, title, fig_si, fo_si, fo_ti_si, fo_te, x_name, y_name="Frequency"):
    if x.ndim == 1:
        pass
    elif (x.ndim == 2) & (x.shape[1] == 1):
        x = x.reshape(-1)
    else:
        raise TypeError("x must be 1d or row vector of 2d, but got {}d".format(x.shape[1]))
    mid_all_sort, interval_sum_sort, left, right = cal_dist(x, bins)

    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title, fontsize=fo_si)
    ax.bar(x=np.arange(bins), height=interval_sum_sort, color="lightcoral", edgecolor="black", linewidth=2,
           label="Distribution")
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    if pos != []:
        mean, std = np.mean(x), np.std(x)
        axes = plt.gca()
        lim_x_min, lim_x_max = axes.get_xlim()
        lim_y_min, lim_y_max = axes.get_ylim()
        lim_x_length, lim_y_length = lim_x_max - lim_x_min, lim_y_max - lim_y_min
        x_loc = pos[0] * lim_x_length + lim_x_min
        y_loc = pos[1] * lim_y_length + lim_y_min
        t = ax.text(x_loc, y_loc, "Mean = {:.5f}\n Std = {:.5f}".format(mean, std), fontsize=fo_te)
        t.set_bbox(dict(facecolor="lightcoral", alpha=0.5, edgecolor="lightcoral"))

    # 使用核函数拟合曲线
    x_t = np.linspace(left, right, 1000)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.7).fit(x.reshape(-1, 1))
    log_density = kde.score_samples(x_t.reshape(-1, 1))
    y_t = np.exp(log_density)
    r_y = np.max(interval_sum_sort) / np.max(y_t)           # 缩放尺度，与柱状图一致
    x_t = np.linspace(0, bins, 1000)
    y_t = y_t * r_y                     # 最终用于绘制的拟合曲线
    ax.plot(x_t, y_t, lw=5, label="Kernel density", c='blue', zorder=2)
    ax.legend(fontsize=fo_si)

    # 求顶点与拐点
    x_ve, y_ve = get_max(x_t, y_t)              # 顶点
    ax.scatter(x_ve, y_ve, s=600, c='yellow', zorder=2, label='Vertex', edgecolor='black', lw=4)
    ax.vlines(x_ve, 0, y_ve, color='black', lw=4, ls='dashed', zorder=1)
    x_ip, y_ip = cal_2d_deri(x_t, y_t)          # 拐点
    ax.scatter(x_ip, y_ip, s=600, c='lightgreen', zorder=2, label='Vertex', edgecolor='black', lw=4)
    ax.vlines(x_ip, 0, y_ip, color='black', lw=4, ls='dashed', zorder=1)

    # 设置刻度
    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=fo_si, labelpad=20)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=fo_si, labelpad=20)

    return fig


save_fig = True
bins = 40
jump = 8
pos = []
title = "HW Distribution"
fig_si = (20, 20)
fo_si = 50
fo_ti_si = 40
fo_te = 50
x_name = "Average production in 3 years ($10 ^{4}$ m$ ^{3}$)"
y_name = "Frequency (%)"

data_ad = '../dataset'
with open(osp.join(data_ad, 'product_all.pkl'), 'rb') as f:
    product_all = pickle.load(f)

vw_info = pd.read_excel(osp.join(data_ad, 'class_info', 'new_cj_su_vwBasicinfo.xls'), sheet_name='常规井')
hw_info = pd.read_excel(osp.join(data_ad, 'class_info', 'new_cj_su_hwBasicinfo.xls'), sheet_name='水平井')
vw_names = vw_info.loc[:, '井名'].values.reshape(-1)
hw_names = hw_info.loc[:, '井名'].values.reshape(-1)

ops = product_all.ops                # 所有油井的油压
dgos = product_all.dgos              # 所有油井的日产气量
cps = product_all.cps                # 所有油井的套压
names = product_all.names            # 所有油井的名称
dates = product_all.dates            # 所有油井的生产日期
num_wells = len(names)

dgo_mean_hw, dgo_mean_vw = [], []
dgos_new_hw, dgos_new_vw = [], []
for i in range(num_wells):
    dgo, name = dgos[i], names[i]
    dgo = dgo[dgo != 0]             # 除去零产量的时刻
    if dgo.shape[0] <= 365 * 3:
        continue
    dgo_mean_i = np.mean(dgo[:(365 * 3)])
    if np.argwhere(vw_names == name).reshape(-1).shape[0] != 0:         # 若为直井
        dgo_mean_vw.append(dgo_mean_i)
        dgos_new_vw.append(dgo)
    elif np.argwhere(hw_names == name).reshape(-1).shape[0] != 0:       # 若为水平井
        dgo_mean_hw.append(dgo_mean_i)
        dgos_new_hw.append(dgo)
    else:
        print("{}，不为直井或者水平井！".format(name))
dgo_mean_hw, dgo_mean_vw = np.array(dgo_mean_hw), np.array(dgo_mean_vw)

fig_dist_hw = plot_dist(dgo_mean_hw, bins, jump, pos, 'HW Distribution', fig_si, fo_si, fo_ti_si, fo_te, x_name, y_name)
fig_dist_vw = plot_dist(dgo_mean_vw, bins, jump, pos, 'VW Distribution', fig_si, fo_si, fo_ti_si, fo_te, x_name, y_name)
if save_fig:
    fig_dist_hw.savefig('../graph/dist_hw.png')
    fig_dist_vw.savefig('../graph/dist_vw.png')

print()
plt.show()
print()
