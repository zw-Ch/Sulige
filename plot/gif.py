import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os.path as osp
import sys
sys.path.append("..")
import func.process as pro


new = pro.NewWellInfo(name="new_WellInfo")
index = 8               # 油井在new.well_name中的索引
well_name = new.well_name[index]
g_address = osp.join("../graph/rnn_one_well_concise", well_name)
re_address = osp.join("../result/rnn_one_well_concise", well_name)

train_true = np.load(osp.join(re_address, "train_true.npy"))[:, 0]
train_predict = np.load(osp.join(re_address, "train_predict.npy"))[:, 0]
test_true = np.load(osp.join(re_address, "test_true.npy"))[:, 0]
test_predict = np.load(osp.join(re_address, "test_predict.npy"))[:, 0]

train_predict_epoch = np.load(osp.join(re_address, "train_predict_epoch.npy"))
test_predict_epoch = np.load(osp.join(re_address, "test_predict_epoch.npy"))

length_train, length_test = train_true.shape[0], test_true.shape[0]     # 训练集和测试集的长度
range_train = np.arange(length_train)                                   # 绘图时，训练集的横轴坐标
range_test = np.arange(length_train, length_train + length_test)        # 绘图时，测试集的横轴坐标

fig, ax = plt.subplots(figsize=(12, 12))


"""
每一天的结果显示
"""
lw = 3
interval = 20
def animate(i):
    ax.clear()
    j = i * interval
    if i <= (length_train // interval):
        l_true, = ax.plot(range_train[0: j], train_true[0: j], lw=lw, alpha=0.5, label="Train original data", color="r")
        l_predict, = ax.plot(range_train[0: j], train_predict[0: j], lw=lw, alpha=0.5, label="Train predict result", color="b")
        p_true, = ax.plot(range_train[j], train_true[j], marker='.', alpha=0.5, markersize=25, color="r")
        p_predict, = ax.plot(range_train[j], train_predict[j], marker='.', alpha=0.5, markersize=25, color="b")
        plt.legend(fontsize=25, loc=1)
        # plt.xlabel("Date (day)", fontsize=35)
        # plt.ylabel("Daily gas production ($10 ^{4}$ m$ ^{3}$)", fontsize=35)
        plt.xlabel("Date (day)", fontsize=35)
        plt.ylabel("Daily gas production ($10 ^{4}$ m$ ^{3}$)", fontsize=35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        return l_true, l_predict, p_true, p_predict
    else:
        j = j - length_train
        l_true_train = ax.plot(range_train, train_true, lw=lw, alpha=0.5, label="Train original data", color="r")
        l_true_predict, = ax.plot(range_train, train_predict, lw=lw, alpha=0.5, label="Train predict result", color="b")
        l_true, = ax.plot(range_test[0: j], test_true[0: j], lw=lw, alpha=0.5, label="Test original data", color="green")
        l_predict, = ax.plot(range_test[0: j], test_predict[0: j], lw=lw, alpha=0.5, label="Test predict result", color="orange")
        p_true, = ax.plot(range_test[j], test_true[j], marker='.', alpha=0.5, markersize=25, color="green")
        p_predict, = ax.plot(range_test[j], test_predict[j], marker='.', alpha=0.5, markersize=25, color="orange")
        plt.legend(fontsize=25, loc=1)
        # plt.xlabel("Date (day)", fontsize=35)
        # plt.ylabel("Daily gas production ($10 ^{4}$ m$ ^{3}$)", fontsize=35)
        plt.xlabel("Date (day)", fontsize=35)
        plt.ylabel("Daily gas production ($10 ^{4}$ m$ ^{3}$)", fontsize=35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        return l_true_train, l_true_predict, l_true, l_predict, p_true, p_predict

ani = FuncAnimation(fig, animate, interval=1, blit=False, repeat=True,
                    frames=(length_train // interval + length_test // interval))
ani.save(osp.join(g_address, "predict_result.gif"), writer=PillowWriter(fps=25))

"""
每一轮的结果显示
"""
# lw = 3
# def animate(i):
#     ax.clear()
#     l_train_true, = ax.plot(range_train, train_true, lw=lw, alpha=0.5, label="Train original data")
#     l_train_predict, = ax.plot(range_train, train_predict_epoch[i, :], lw=lw, alpha=0.5, label="Train predict result")
#     l_test_true, = ax.plot(range_test, test_true, lw=lw, alpha=0.5, label="Test original data")
#     l_test_predict, = ax.plot(range_test, test_predict_epoch[i, :], lw=lw, alpha=0.5, label="Test predict result")
#     plt.legend(fontsize=25, loc=1)
#     plt.xlabel("Date (day)", fontsize=35)
#     plt.ylabel("Daily gas production ($10 ^{4}$ m$ ^{3}$)", fontsize=35)
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)
#     plt.title("{} Epoch".format(i + 1), fontsize=35)
#     return l_train_true, l_train_predict, l_test_true, l_test_predict
#
# ani = FuncAnimation(fig, animate, interval=1, blit=True, repeat=True, frames=10)
# ani.save(osp.join(g_address, "epoch_result.gif"), writer=PillowWriter(fps=5))
