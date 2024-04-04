"""
画图的
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['PROJ_LIB'] = '<path to anaconda>/anaconda3/share/proj'
from datetime import datetime
from sklearn.neighbors import KernelDensity
from mpl_toolkits.basemap import Basemap
from scipy.spatial import Delaunay
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.patches import Rectangle


def cal_dist(x, bins):
    x_out, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_out = pd.DataFrame(x_out)
    x_out_vc = pd.DataFrame(x_out).value_counts()
    interval = x_out_vc.index.tolist()
    interval_sum = x_out_vc.values
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
    interval_sum_sort = interval_sum[sort_index]
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


def plot_delaunay(locs, z, z_name, fig_size=(12, 12), fo_si=35, fo_ti_si=25, k=3):
    """
    Plot Delaunay Figure

    :param locs: Location
    :param z: Z axis
    :param z_name:
    :param fig_size:
    :param fo_si:
    :param fo_ti_si:
    :param k: Interpolation times
    :return:
    """

    def get_idx(x_, y_):
        return np.argwhere((x == x_) & (y == y_)).reshape(-1)[0]

    def cal():
        cx, cy = center[index][0], center[index][1]
        x1, y1, x2, y2, x3, y3 = sim[0][0], sim[0][1], sim[1][0], sim[1][1], sim[2][0], sim[2][1]
        idx_1, idx_2, idx_3 = get_idx(x1, y1), get_idx(x2, y2), get_idx(x3, y3)
        z_1, z_2, z_3 = z[idx_1], z[idx_2], z[idx_3]
        z_mean = np.mean([z_1, z_2, z_3])
        return cx, cy, z_mean

    x, y = locs[:, 0], locs[:, 1]
    tri = Delaunay(locs)
    center = np.sum(locs[tri.simplices], axis=1) / 3.0      # 每个三角形的重心
    locs_new, locs_ori = locs, locs
    z_new = z.tolist()

    while k != 0:
        for index, sim in enumerate(locs[tri.simplices]):
            cx_, cy_, z_ = cal()
            point_one = np.array([cx_, cy_]).reshape(1, -1)  # 增加的一个节点，即中心点
            locs_new = np.concatenate((locs_new, point_one), axis=0)
            z_new.append(z_)  # 增加的一个深度
        locs = locs_new
        z = z_new
        tri = Delaunay(locs)
        center = np.sum(locs[tri.simplices], axis=1) / 3.0
        x, y = locs[:, 0], locs[:, 1]
        k = k - 1

    z = np.array(z)
    color = []
    for index, sim in enumerate(locs[tri.simplices]):
        cx_, cy_, z_ = cal()
        color.append(z_)
    color = np.array(color)

    fig = plt.figure(figsize=fig_size)
    plt.tripcolor(locs[:, 0], locs[:, 1], tri.simplices.copy(), facecolors=color, edgecolors='none',
                  cmap=plt.cm.Spectral_r)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fo_ti_si)
    plt.tick_params(labelbottom='off', labelleft='off', left='off', right='off', bottom='off', top='off')
    plt.scatter(locs_ori[:, 0], locs_ori[:, 1], color='r', s=15)    # 将油井的位置标出
    plt.title(z_name, fontsize=fo_si, pad=20)
    plt.xlabel("X", fontsize=fo_si, labelpad=10)
    plt.ylabel("Y", fontsize=fo_si, labelpad=10)
    plt.xticks(fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    return fig


def plot_date(dates):
    """
    Plot Date of different wells

    :param dates: List of numpy.ndarray
    """
    dates_convert = [(datetime.strptime(date[0], '%Y/%m/%d'), datetime.strptime(date[-1], '%Y/%m/%d')) for date in dates]
    dates_convert = dates_convert[:10]

    min_date = min(date for pair in dates_convert for date in pair)
    max_date = max(date for pair in dates_convert for date in pair)

    return None


def plot_map(locs, bins=8):
    lat, lon = locs[:, 0], locs[:, 1]
    lat_bounds = [np.min(lat), np.max(lat)]
    lon_bounds = [np.min(lon), np.max(lon)]
    m = Basemap(projection='cyl',
                llcrnrlon=lon_bounds[0],
                llcrnrlat=lat_bounds[0],
                urcrnrlon=lon_bounds[1],
                urcrnrlat=lat_bounds[1])
    # parallels = np.linspace(lat_bounds[0], lat_bounds[1], bins)
    # meridians = np.linspace(lon_bounds[0], lon_bounds[1], bins)
    # m.drawcoastlines()
    # m.drawmapboundary()
    x, y = m(lon, lat)
    m.scatter(x, y)
    pass
