"""
画图的
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


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

    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=fo_si, labelpad=20)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=fo_si, labelpad=20)

    # fit curve
    x_plot = np.linspace(left, right, 1000)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x.reshape(-1, 1))
    log_density = kde.score_samples(x_plot.reshape(-1, 1))
    y_plot = np.exp(log_density)
    r_y = np.max(interval_sum_sort) / np.max(y_plot)
    x_t = np.linspace(0, bins, 1000)
    ax.plot(x_t, y_plot * r_y, lw=5, label="Kernel density", c='blue', zorder=2)
    ax.legend(fontsize=fo_si)

    return fig
