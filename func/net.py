import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn


def cal_rmse_one_arr(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def cal_r2_one_arr(true, pred):
    corr_matrix = np.corrcoef(true, pred)
    corr = corr_matrix[0, 1]
    r2 = corr ** 2
    return r2


class GNN(nn.Module):
    def __init__(self, gnn_style, adm_style, k, num_nodes, device):
        super().__init__()
        self.gnn_style = gnn_style
        self.gnn1 = get_gnn(gnn_style, 1, 16)
        self.gnn2 = get_gnn(gnn_style, 16, 1)
        self.ei1, self.ew1 = get_edge_info(k, num_nodes, adm_style, device)
        self.ei1, self.ew1 = get_edge_info(k, num_nodes, adm_style, device)

    def forward(self, x):
        h = run_gnn(self.gnn_style, self.gnn1, x, self.ei1, self.ew1)
        h = run_gnn(self.gnn_style, self.gnn2, h, self.ei2, self.ew2)
        return h


def get_gnn(gnn_style, in_dim, out_dim):
    if gnn_style == "gcn":
        return gnn.GCNConv(in_dim, out_dim)
    elif gnn_style == "cheb":
        return gnn.ChebConv(in_dim, out_dim, K=1)
    elif gnn_style == "gin":
        return gnn.GraphConv(in_dim, out_dim)
    elif gnn_style == "graphsage":
        return gnn.SAGEConv(in_dim, out_dim)
    elif gnn_style == "tag":
        return gnn.TAGConv(in_dim, out_dim)
    elif gnn_style == "sg":
        return gnn.SGConv(in_dim, out_dim)
    elif gnn_style == "appnp":
        return gnn.APPNP(K=2, alpha=0.5)
    elif gnn_style == "arma":
        return gnn.ARMAConv(in_dim, out_dim)
    elif gnn_style == "cg":
        return gnn.CGConv(in_dim)
    elif gnn_style == "unimp":
        return gnn.TransformerConv(in_dim, out_dim)
    elif gnn_style == "edge":
        layer = nn.Linear(2 * in_dim, out_dim)
        return gnn.EdgeConv(layer)
    elif gnn_style == "gan":
        return gnn.GATConv(in_dim, out_dim)
    elif gnn_style == "mf":
        return gnn.MFConv(in_dim, out_dim)
    elif gnn_style == "resgate":
        return gnn.ResGatedGraphConv(in_dim, out_dim)
    else:
        raise TypeError("Unknown type of gnn_style!")


def run_gnn(gnn_style, gnn, x, ei, ew):
    if gnn_style in ["gcn", "cheb", "sg", "appnp", "tag"]:
        return gnn(x, ei, ew)
    elif gnn_style in ["unimp", "gan"]:
        batch_size = x.shape[0]  # 批量数量
        h_all = None
        for i in range(batch_size):  # 将每个样本输入图神经网络后，将每个输出结果拼接
            x_one = x[i, :, :]
            h = gnn(x_one, ei)
            h = h.unsqueeze(0)
            if h_all is None:
                h_all = h
            else:
                h_all = torch.cat((h_all, h), dim=0)
        return h_all
    else:
        return gnn(x, ei)


def get_edge_info(k, num_nodes, adm_style, device):
    if adm_style == "ts_un":
        adm = ts_un(num_nodes, k)
    elif adm_style == "tg":
        adm = tg(num_nodes)
    else:
        raise TypeError("Unknown type of adm_style!")
    edge_index, edge_weight = tran_adm_to_edge_index(adm)
    edge_index = edge_index.to(device)
    return edge_index, nn.Parameter(edge_weight)


def ts_un(n, k):
    adm = np.zeros(shape=(n, n))
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    else:
        for i in range(n):
            if i < (n - k):
                for k_one in range(1, k + 1):
                    adm[i, i + k_one] = 1.
            else:
                for k_one in range(1, k + 1):
                    if (k_one + i) >= n:
                        pass
                    else:
                        adm[i, i + k_one] = 1.
    adm = (adm.T + adm) / 2
    # adm = adm * 0.5
    return adm


def tg(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i + 1, i] = 1
    adm[0, m - 1] = 1
    adm = adm * 0.5
    return adm


def tran_adm_to_edge_index(adm):
    u, v = np.nonzero(adm)
    num_edges = u.shape[0]
    edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
    edge_weight = np.zeros(shape=u.shape)
    for i in range(num_edges):
        edge_weight_one = adm[u[i], v[i]]
        edge_weight[i] = edge_weight_one
    edge_index = torch.from_numpy(edge_index).long()
    edge_weight = torch.from_numpy(edge_weight).float()
    return edge_index, edge_weight
