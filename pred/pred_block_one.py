import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import os.path as osp
import sys
sys.path.append('..')
import func.well as well
import func.draw as draw
import func.net as net


device = "cuda:0" if torch.cuda.is_available() else "cpu"
root = "../dataset"
block_name = "47（2）"
lx = 120
ly = 12
train_ratio = 0.75

block = well.Block(root, block_name)
block.remain_length([1000, 6000])
print(block)

data = block.get_data()
train_loader, test_loader = well.get_loader(data, train_ratio, lx, ly)

model = net.SimpleLSTM(lx, 32, ly).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    train_true, train_pred = [], []
    test_true, test_pred = [], []

    model.train()
    for i, (train_x, train_y) in enumerate(train_loader):
        train_x, train_y = train_x.to(device), train_y.to(device)

        optimizer.zero_grad()
        train_out = model(train_x)
        train_loss = criterion(train_out, train_y)
        train_loss.backward()
        optimizer.step()

        train_true.append(train_y[0, :, -1].detach().cpu().numpy())
        train_pred.append(train_out[0, :, -1].detach().cpu().numpy())

    model.eval()
    for i, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)

        test_out = model(test_x)
        test_loss = criterion(test_out, test_y)

        test_true.append(test_y[0, :, -1].detach().cpu().numpy())
        test_pred.append(test_out[0, :, -1].detach().cpu().numpy())

    r2_train = well.cal_r2(train_true, train_pred)
    rmse_train = well.cal_rmse(train_true, train_pred)
    r2_test = well.cal_r2(test_true, test_pred)
    rmse_test = well.cal_rmse(test_true, test_pred)
    print(f'Epoch [{epoch + 1}/{epochs}], R2 Test: {r2_test:.4f}, R2 Train: {r2_train:.4f}, RMSE Train: {rmse_train:.4f}, RMSE Test: {rmse_test:.4f}')

# plt.figure()
# plt.plot(well.data)

# fig_delaunay = draw.plot_delaunay(loc, length, "Length")

plt.show()
print()
