import pandas as pd
from scipy.spatial import Delaunay
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def remain_range(x_ran, y_ran, x, y, z):
    x_min, x_max, y_min, y_max = x_ran[0], x_ran[1], y_ran[0], y_ran[1]
    idx = np.argwhere((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)).reshape(-1)
    x, y, z = x[idx], y[idx], z[idx]
    return x, y, z


def get_block(root, style, block, attr_name):
    if style == "直井":
        info = pd.read_excel(osp.join(root, "new_cj_su_vwBasicinfo.xls"))
    elif style == "水平井":
        info = pd.read_excel(osp.join(root, "new_cj_su_hwBasicinfo.xls"))
    else:
        raise TypeError("Unknown type of style, must be '直井' or '水平井'！")
    info = info[info['区块'] == block]
    wells_info = info.loc[:, "井名"].values.reshape(-1)
    files = ["new_cj_su_hwFPDCv1.xls", "new_cj_su_hwFV1.xls", "new_cj_su_vwFPDCv1.xls", "new_cj_su_vwFV1.xls"]
    x, y, z = [], [], []
    for file in files:
        data = pd.read_excel(osp.join(root, file))
        idx = np.argwhere(data["井名"].isin(wells_info).values).reshape(-1)
        if idx.shape[0] != 0:
            data = data.iloc[idx, :]
            x_ = data.loc[:, "井口纵坐标X"].values
            y_ = data.loc[:, "井口横坐标Y"].values
            z_ = data.loc[:, attr_name].values
            if len(x) == 0:
                x, y, z = x_, y_, z_
            else:
                x = np.concatenate((x, x_))
                y = np.concatenate((y, y_))
                z = np.concatenate((z, z_))
    return x, y, z


block_in_vw = ["桃2", "苏47", "苏48", "苏120", "苏14", "苏47"]
block_in_hw = ["桃2", "苏47", "苏48", "苏120", "苏14"]

fig_si = (16, 16)
fo_si = 35
fo_bar_si = 20
fo_ti_si = 15

root = "../dataset/class_info/"
style = "直井"                # 可选："直井", "水平井"
block = "苏47"                # 可选：若为"直井"，从block_in_vw中挑选；若为"水平井"，从block_in_hw中挑选
attr_name = "含油气饱和度"     # 可选：厚度、电阻率、声波时差、岩石密度、泥质含量、总孔隙度、渗透率、含油气饱和度

x, y, z = get_block(root, style, block, attr_name)

# 如果有的油井位置过于遥远（异常），通过设置坐标范围，仅保留部分油井
x_ran = np.array([4.17, 4.23]) * 1e6
y_ran = np.array([1.92, 1.93]) * 1e7
x, y, z = remain_range(x_ran, y_ran, x, y, z)

points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)   # 坐标

# 初始化
tri = Delaunay(points)
center = np.sum(points[tri.simplices], axis=1) / 3.0        # 每个三角形的重心
points_new = points
points_ori = points             # 原始的油井位置坐标
z_new = z.tolist()

k = 3               # 内插次数
while k != 0:
    for index, sim in enumerate(points[tri.simplices]):
        cx, cy = center[index][0], center[index][1]             # 区域的中心点
        x1, y1 = sim[0][0], sim[0][1]           # 区域周围的三个点
        x2, y2 = sim[1][0], sim[1][1]
        x3, y3 = sim[2][0], sim[2][1]
        idx_1 = np.argwhere((x == x1) & (y == y1)).reshape(-1)[0]       # 找到三个点对应的索引
        idx_2 = np.argwhere((x == x2) & (y == y2)).reshape(-1)[0]
        idx_3 = np.argwhere((x == x3) & (y == y3)).reshape(-1)[0]
        z_1, z_2, z_3 = z[idx_1], z[idx_2], z[idx_3]    # 三个点对应的变量值
        z_one = np.mean([z_1, z_2, z_3])
        point_one = np.array([cx, cy]).reshape(1, -1)             # 增加的一个节点，即中心点
        points_new = np.concatenate((points_new, point_one), axis=0)
        z_new.append(z_one)             # 增加的一个深度
    points = points_new
    z = z_new
    tri = Delaunay(points)
    center = np.sum(points[tri.simplices], axis=1) / 3.0
    x, y = points[:, 0], points[:, 1]

    k = k - 1

z = np.array(z)
color = []
for index, sim in enumerate(points[tri.simplices]):
    cx, cy = center[index][0], center[index][1]
    x1, y1 = sim[0][0], sim[0][1]
    x2, y2 = sim[1][0], sim[1][1]
    x3, y3 = sim[2][0], sim[2][1]
    idx_1 = np.argwhere((x == x1) & (y == y1)).reshape(-1)[0]
    idx_2 = np.argwhere((x == x2) & (y == y2)).reshape(-1)[0]
    idx_3 = np.argwhere((x == x3) & (y == y3)).reshape(-1)[0]
    z_1, z_2, z_3 = z[idx_1], z[idx_2], z[idx_3]
    z_mean = np.mean([z_1, z_2, z_3])
    color.append(z_mean)
color = np.array(color)

plt.figure(figsize=(12, 12))
plt.tripcolor(points[:, 0], points[:, 1], tri.simplices.copy(), facecolors=color, edgecolors='none',
              cmap=plt.cm.Spectral_r)
cbar = plt.colorbar()          # 增加色标柱
# cbar.set_label(attr_name, fontsize=fo_bar_si, labelpad=10)
cbar.ax.tick_params(labelsize=fo_ti_si)

plt.tick_params(labelbottom='off', labelleft='off', left='off', right='off', bottom='off', top='off')
plt.scatter(points_ori[:, 0], points_ori[:, 1], color='r', s=15)            # 将油井的位置标出
plt.title(attr_name, fontsize=fo_si, pad=20)
plt.xlabel("井口纵坐标X", fontsize=fo_si, labelpad=10)
plt.ylabel("井口横坐标Y", fontsize=fo_si, labelpad=10)
plt.xticks(fontsize=fo_ti_si)
plt.yticks(fontsize=fo_ti_si)

print()
plt.show()
print()
