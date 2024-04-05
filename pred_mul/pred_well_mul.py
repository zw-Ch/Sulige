import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
sys.path.append('..')
import func.well as well
import func.draw as draw


root = "../dataset"
# block_name = "120（1）"
# block = well.Block(root, block_name)

locs, lengths, dates = [], [], []
x_ran = np.array([4.22, 4.26]) * 1e6
y_ran = np.array([1.91, 1.928]) * 1e7
length_ran = [3000, 6000]

for block_name in os.listdir(osp.join(root, "product_data")):
    block_name = block_name.split('.')[0]
    block = well.Block(root, block_name)

    block.remain_loc(x_ran, y_ran)
    block.remain_length(length_ran)

    locs.append(block.get_loc())
    lengths.append(block.get_length())
    dates.append(block.get_date())

locs = np.vstack(locs)
dates = [date for one in dates for date in one]

# fig = plt.Figure()
# plt.scatter(locs[:, 0], locs[:, 1])

# fig_delaunay = draw.plot_delaunay(locs, lengths, "Length")
fig_date = draw.plot_date(dates)

plt.show()
print()
