import numpy as np
import torch
import os.path as osp
import pickle

from torch_cluster import knn
from pygsp.graphs import Graph
from scipy.sparse import csr_matrix
from torch_geometric.nn import knn_graph


# 自定义图结构
class SelfGraph(Graph):
    def __init__(self, xy, a, vertex_size=30, labels=None, gtype='self_graph', edge_index=False, **kwargs):
        xy = xy                         # 节点坐标
        A = a                           # 邻接矩阵
        vertex_size = vertex_size       # 节点大小
        if edge_index:                  # 计算edge_index形式
            self.edge_index, self.edge_weight = tran_adm_to_edge_index(adm=a)
        self.labels = labels            # 节点标签
        self.adm = a

        # 设定绘图范围，比节点坐标的最大值（最小值）大（小）1；设定节点大小
        y_max = np.min(xy[:, 0]) - 1
        y_min = np.max(xy[:, 0]) + 1
        x_max = np.max(xy[:, 1]) - 1
        x_min = np.min(xy[:, 1]) + 1
        plotting = {"limits": np.array([y_min, y_max, x_min, x_max]),
                    "vertex_size": vertex_size}

        super(SelfGraph, self).__init__(W=A, coords=xy, gtype=gtype, plotting=plotting, **kwargs)


# 由节点坐标，通过k近邻生成图
class KnnGraph(SelfGraph):
    def __init__(self, xy, k=2, loop=False, style="adm", norm=False, arr=True, s=30, labels=None, undirected=True):
        self.style = style          # 邻接矩阵 or csr_matrix稀疏矩阵
        self.norm = norm            # 拉普拉斯矩阵是否归一化
        self.arr = arr              # 是否计算邻接矩阵、拉普拉斯矩阵的array格式
        num_nodes = xy.shape[0]     # 节点数量

        # 通过节点坐标，获得k近邻图的邻接矩阵
        if isinstance(xy, np.ndarray):              # 若输入为数组
            xy_tensor = torch.from_numpy(xy)
        elif torch.is_tensor(xy):                   # 若输入为张量
            xy_tensor = xy
            xy = xy.detach().cpu().numpy()
        batch = torch.tensor([0] * num_nodes)
        edge_index = knn_graph(xy_tensor, k=k, batch=batch, loop=loop)
        if undirected:      # 变成无向图的形式
            edge_index = tran_edge_index_to_undirected(edge_index)
        self.edge_index = edge_index

        if style == "csr":      # 邻接矩阵为稀疏矩阵形式（当节点数量过多时，需要花费相当多时间）
            edge_index = edge_index.detach().numpy()
            row = []
            for i in range(num_nodes):
                row_one = [i] * num_nodes
                row = row + row_one                             # 行指标
            col = np.arange(num_nodes).tolist() * num_nodes     # 列指标
            row_col = np.hstack([np.array(row).reshape(-1, 1), np.array(col).reshape(-1, 1)])

            num_edges = edge_index.shape[1]     # 边数量
            data = np.array([0] * num_nodes ** 2)
            j_all = []
            for i in range(num_edges):
                edge = edge_index[:, i]
                for j in range(num_edges ** 2):
                    row_col_one = row_col[j, :]
                    if (row_col_one == edge).all():
                        j_all.append(j)
                        break
            data[j_all] = 1
            matrix = csr_matrix((data.tolist(), (row, col)), shape=(num_nodes, num_nodes))        # 稀疏矩阵形式
        elif style == "adm":
            matrix = tran_edge_index_to_adm(edge_index=edge_index)
            if self.arr:
                self.adm = matrix
                self.lam = get_laplacian_matrix(adm=self.adm, norm=norm)
        else:
            raise TypeError("style must be 'adm' or 'csr', but got {}".format(style))

        gtype = "knn_graph"
        super(KnnGraph, self).__init__(xy=xy, a=matrix, gtype=gtype, vertex_size=s, labels=labels)


# 让edge_index变成无向图的形式
def tran_edge_index_to_undirected(edge_index):
    edge_index = edge_index.detach().cpu().numpy()
    num_edges = edge_index.shape[1]  # 此时，边的数量
    u, v = edge_index[0, :], edge_index[1, :]
    edge_all = []
    for i in range(num_edges):
        u_one, v_one = u[i], v[i]
        edge_one = (u_one, v_one)
        edge_all.append(edge_one)
        edge_one_pair = (v_one, u_one)
        edge_all.append(edge_one_pair)
        edge_all = list(set(edge_all))

    num_edges_true = len(edge_all)      # 边的真实数量
    u_all, v_all = [], []
    for i in range(num_edges_true):
        edge_one = edge_all[i]
        u_one, v_one = edge_one[0], edge_one[1]
        u_all.append(u_one)
        v_all.append(v_one)
    u_all, v_all = np.array(u_all).reshape(1, -1), np.array(v_all).reshape(1, -1)
    edge_index_true = np.vstack([u_all, v_all])
    edge_index_true = torch.from_numpy(edge_index_true)

    return edge_index_true


# 将临界矩阵的类型由edge_index转化为array
def tran_edge_index_to_adm(edge_index, num_nodes_truth=None):
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index_r1, edge_index_r2 = edge_index[0, :].reshape(-1), edge_index[1, :].reshape(-1)

    # 节点的数量
    edge_index_r1_r2 = np.hstack([edge_index_r1, edge_index_r2])
    if num_nodes_truth is not None:         # 当图不是连接图时（存在孤立节点时）
        num_nodes = num_nodes_truth
        nodes = np.arange(num_nodes)
    else:
        nodes = np.unique(edge_index_r1_r2)
        num_nodes = nodes.shape[0]
    adm = np.zeros(shape=(num_nodes, num_nodes))

    for i in range(num_nodes):
        node = nodes[i]
        neighbor_nodes_index = np.where(edge_index_r1 == node)[0].tolist()
        neighbor_nodes = edge_index_r2[neighbor_nodes_index].tolist()
        adm[i, neighbor_nodes] = 1.

    return adm


# 将邻接矩阵的类型由array转化为edge_index
def tran_adm_to_edge_index(adm):
    if isinstance(adm, np.ndarray):         # array数组类型
        u, v = np.nonzero(adm)
        num_edges = u.shape[0]          # 边的数量
        edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])        # 边索引
        edge_weight = np.zeros(shape=u.shape)           # 初始化边权重
        for i in range(num_edges):
            edge_weight_one = adm[u[i], v[i]]
            edge_weight[i] = edge_weight_one
        edge_index = torch.from_numpy(edge_index.astype(np.int64))
        edge_weight = torch.from_numpy(edge_weight.astype(np.float32))
        return edge_index, edge_weight
    elif torch.is_tensor(adm):              # pytorch张量类型
        u, v = torch.nonzero(adm, as_tuple=True)
        num_edges = u.shape[0]          # 边的数量
        edge_index = torch.cat((u.view(1, -1), v.view(1, -1)))
        edge_weight = torch.zeros(size=u.shape)
        for i in range(num_edges):
            edge_weight[i] = adm[u[i], v[i]]
        return edge_index, edge_weight


def get_laplacian_matrix(adm, norm=False):                 # 计算拉普拉斯矩阵
    degree_mat = get_degree_matrix(adm=adm, not_matrix=False)
    laplacian_matrix = degree_mat - adm
    if norm:
        degree_mat_inv_sqrt = np.power(np.diagonal(degree_mat), -0.5)
        laplacian_matrix = degree_mat_inv_sqrt * laplacian_matrix * degree_mat_inv_sqrt
    return laplacian_matrix


def get_degree_matrix(adm, not_matrix=False):              # 计算度矩阵
    sum_all = []
    m = adm.shape[0]
    for i in range(m):
        matrix_one = adm[i, :]
        sum_one = matrix_one.sum()
        sum_all.append(sum_one)
    sum_all = np.array(sum_all).reshape(-1)
    degree_mat = np.diag(sum_all)
    if not_matrix:
        degree_mat = np.diagonal(degree_mat)
    return degree_mat
