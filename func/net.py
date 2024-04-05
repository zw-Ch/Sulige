import numpy as np
import math
import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class GNN(nn.Module):
    def __init__(self, gnn_style, adm_style, k, num_nodes, device):
        super().__init__()
        self.gnn_style = gnn_style
        self.k = k
        self.gnn1 = get_gnn(gnn_style, 1, 16)
        self.gnn2 = get_gnn(gnn_style, 16, 1)
        self.ei1, self.ew1 = get_edge_info(k, num_nodes, adm_style, device)
        self.ei1, self.ew1 = get_edge_info(k, num_nodes, adm_style, device)

    def forward(self, x):
        h = run_gnn(self.gnn_style, self.gnn1, x, self.k)
        h = run_gnn(self.gnn_style, self.gnn2, h, self.k)
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


def run_gnn(gnn_style, gnn_, x, adm_style, k, device):
    edge_index = get_edge_info(k, x.size(1), adm_style, device)
    if gnn_style in ["gcn", "cheb", "sg", "appnp", "tag"]:
        return gnn_(x, edge_index)
    elif gnn_style in ["unimp", "gan"]:
        batch_size = x.shape[0]  # 批量数量
        h_all = None
        for i in range(batch_size):  # 将每个样本输入图神经网络后，将每个输出结果拼接
            x_one = x[i, :, :]
            h = gnn_(x_one, edge_index)
            h = h.unsqueeze(0)
            if h_all is None:
                h_all = h
            else:
                h_all = torch.cat((h_all, h), dim=0)
        return h_all
    else:
        return gnn_(x, edge_index)


def get_edge_info(k, num_nodes, adm_style, device):
    if adm_style == "ts_un":
        u, v = [], []
        for i in range(num_nodes):
            left = max(0, i - k)
            right = min(num_nodes - 1, i + k)
            v_ = np.concatenate((np.arange(left, i), np.arange(i + 1, right + 1)))
            u_ = np.full(v_.shape, i, dtype=int)
            u.extend(u_), v.extend(v_)
        edge_index = torch.tensor([u, v], dtype=torch.long)
    # elif adm_style == "tg":
    #     adm = tg(num_nodes)
    else:
        raise TypeError("Unknown type of adm_style!")
    # edge_index, edge_weight = tran_adm_to_edge_index(adm)
    # edge_index = edge_index.to(device)
    return edge_index.to(device)


def ts_un(n, k):
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    adm = np.zeros((n, n))
    for i in range(n - 1):
        adm[i, i + 1: i + k + 1] = 1
    adm = (adm.T + adm) / 2
    return adm


def tg(m):
    adm = np.zeros((m, m))
    adm[1:, :-1] = 1
    adm[0, -1] = 1
    adm *= 0.5
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


# def generate_square_subsequent_mask(sq_len):
#     """
#     Generate Mask Matrix for Transformer Attention Mechanism
#
#     :param sq_len: Length of sequence (int)
#     :return: torch.Tensor
#     """
#     mask = (torch.triu(torch.ones(sq_len, sq_len)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.relu(self.drop(out))
        out = self.fc(out)
        return out


class Trans(nn.Module):
    def __init__(self, d_model, out_dim, n_head, num_layers, length_max, adm_style, gnn_style, k, device):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head, batch_first=True), num_layers)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.decoder = nn.Linear(d_model, out_dim)
        self.position_enc = PositionalEncoding(d_model, length_max)

        self.adm_style, self.gnn_style, self.k, self.device = adm_style, gnn_style, k, device
        self.gnn1 = get_gnn(gnn_style, d_model, 32)
        self.gnn2 = get_gnn(gnn_style, 32, d_model)

    def forward(self, src):
        src = self.position_enc(src)
        memory = self.encoder(src)

        h = self.relu(self.drop(memory))
        h = run_gnn(self.gnn_style, self.gnn1, h, self.adm_style, self.k, self.device)
        h = self.relu(self.drop(h))
        h = run_gnn(self.gnn_style, self.gnn2, h, self.adm_style, self.k, self.device)
        h = self.relu(self.drop(h))

        output = self.decoder(h)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length_max):
        super().__init__()
        pe = torch.zeros(length_max, d_model)
        position = torch.arange(0, length_max).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x