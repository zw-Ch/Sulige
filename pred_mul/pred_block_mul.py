import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
sys.path.append('..')
import func.well as well
import func.draw as draw
import func.net as net


# ["LSTM", "Trans"]
model_style = "Trans"
device = "cuda:1" if torch.cuda.is_available() else "cpu"
root = "../dataset"
lx = 120
ly = 12
train_ratio = 0.75
epochs = 50

"""
LSTM Parameter
"""
hid_dim = 64

"""
TransformerModel Parameter
"""
n_head = 8
num_layers = 4
gnn_style = "gcn"
adm_style = "ts_un"
k = 1

blocks_name = ['120（1）']
blocks = well.Blocks(root, blocks_name)
blocks.remove_data_zero(0.001)
length_min = 2000
length_max = 6000
blocks.remain_length([length_min, length_max])

data = blocks.get_data()
wells_name = blocks.wells_name
train_loader, test_loader = well.get_loader(data, wells_name, train_ratio, lx, ly)

if model_style == "LSTM":
    model = net.SimpleLSTM(lx, hid_dim, ly).to(device)
elif model_style == "Trans":
    model = net.Trans(lx, ly, n_head, num_layers, length_max, adm_style, gnn_style, k, device).to(device)
else:
    raise TypeError("Unknown type of 'model_style'")
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

for epoch in range(epochs):
    train_true, train_pred = [], []
    test_true, test_pred = [], []
    train_names, test_names = [], []

    model.train()
    for i, (train_x, train_y, train_name) in enumerate(train_loader):
        train_x, train_y = train_x.to(device), train_y.to(device)

        optimizer.zero_grad()
        train_out = model(train_x)
        train_loss = criterion(train_out, train_y)
        train_loss.backward()
        optimizer.step()

        train_true.append(train_y[0, :, -1].detach().cpu().numpy())
        train_pred.append(train_out[0, :, -1].detach().cpu().numpy())
        train_names.append(train_name[0])

    model.eval()
    for i, (test_x, test_y, test_name) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)

        test_out = model(test_x)
        test_loss = criterion(test_out, test_y)

        test_true.append(test_y[0, :, -1].detach().cpu().numpy())
        test_pred.append(test_out[0, :, -1].detach().cpu().numpy())
        test_names.append(test_name[0])

    r2_train = well.cal_r2(train_true, train_pred)
    rmse_train = well.cal_rmse(train_true, train_pred)
    r2_test = well.cal_r2(test_true, test_pred)
    rmse_test = well.cal_rmse(test_true, test_pred)
    print(f'Epoch [{epoch + 1}/{epochs}], R2 Test: {r2_test:.4f}, R2 Train: {r2_train:.4f}, RMSE Train: {rmse_train:.4f}, RMSE Test: {rmse_test:.4f}')

for i in range(5):
    fig = plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(test_true[i], label="true")
    plt.plot(test_pred[i], label="pred")
    plt.title(test_names[i])
    plt.legend()

print()
plt.show()
print()
