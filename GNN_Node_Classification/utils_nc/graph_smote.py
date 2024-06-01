import ast
import copy
import os
import random
from os.path import join

import dgl
import numpy as np
import pandas as pd
import torch.nn.functional as F
import math
import torch.optim as optim

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.spatial.distance import pdist, squareform


class Decoder(Module):
    """
    根据特征生成邻接矩阵，对应论文中使用权重点积来生成边信息
    adopt a vanilla design, weighted inner production
    """

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))  # 权重矩阵
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))
        return adj_out


def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=1):
    """
    使用 smote 进行节点插值生成节点

    :param embed：原始节点的嵌入矩阵。
    :param labels：节点的标签。
    :param idx_train：训练集的节点索引。
    :param adj：邻接矩阵（可选）。
    :param portion：从每个类别选择的节点的比例, default 1
    :param im_class_num：进行过采样的类别数量, default 1

    # 示例数据
    embed = torch.randn(10, 5)  # 10个节点，每个节点有5维嵌入
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
    idx_train = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    adj = torch.randint(0, 2, (10, 10)).float()
    形状为 (10, 10) 的张量，张量中的元素是从 0 到 1（包括 0 和 1）的随机整数，并将这些整数转换为浮点数
    """
    # 通过将训练集样本总数除以类别总数，可以得到每个类别平均分配的样本数量
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new = None

    for i in range(im_class_num):
        # 针对需要进行采样的类别分别进行采样，我们这里设置为 1 即可
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        # 计算需要采样的个数和轮数
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])  # 如果没有设置就采样达到平均值
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            # 寻找最近邻插值
            # pdist 计算每对节点之间的成对距离，返回一个一维距离数组。
            # squareform 将这个一维距离数组转换为二维距离矩阵，使得 distance[i, j] 表示第 i 个节点和第 j 个节点之间的距离。
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            # 用一个很大的数填充距离矩阵的对角线，防止在寻找最近邻时把自己作为最近邻。
            np.fill_diagonal(distance, distance.max() + 100)
            # 返回每个节点的最近邻的索引，即在距离矩阵中每行最小值的位置索引。
            idx_neighbor = distance.argmin(axis=-1)
            # with the nearest neighbor, generate synthetic nodes
            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)
            # 更新数据
            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)
            # 如果有邻接矩阵，更新邻接矩阵
            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)
    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]
        return embed, labels, idx_train, new_adj.detach()
    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt):
    """
    计算所有非合成节点的边预测损失，均方误差（MSE）损失

    :param adj_rec: 使用 edge generator 预测之后的邻接矩阵
    :param adj_tgt: 原始的邻接矩阵
    """
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2
    # 负权重用于惩罚未观察到的边。边的数量除以不存在的边的数量。
    if total_num == edge_num:
        neg_weight = 1
    else:
        neg_weight = edge_num / (total_num - edge_num)
    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight
    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)
    return loss


def get_graph_files(dataset_path, mode, step: int):
    """
    获取图数据保存文件

    :return: [(nodes.tsv, edges.tsv), ...]
    """
    graph_path = join(dataset_path, f'model_dataset_{str(step)}')
    graph_files = []
    if mode == 'train':
        if os.path.exists(join(graph_path, 'train')):
            model_list = os.listdir(join(graph_path, 'train'))
        else:
            model_list = []
    elif mode == 'valid':
        if os.path.exists(join(graph_path, 'valid')):
            model_list = os.listdir(join(graph_path, 'valid'))
        else:
            model_list = []
    else:
        if os.path.exists(join(graph_path, 'test')):
            model_list = os.listdir(join(graph_path, 'test'))
        else:
            model_list = []
    for model_dir in model_list:
        # if not model_dir.startswith('my_mylyn_5834_0'):
        #     continue
        # print(model_dir)
        if not os.path.exists(join(graph_path, mode, model_dir, 'nodes.tsv')):
            continue
        graph_files.append(
            (join(graph_path, mode, model_dir, 'nodes.tsv'), join(graph_path, mode, model_dir, 'edges.tsv')))
    return graph_files


# Load data
def generate_graph_data(mode, step=1):
    graphs = []
    dataset_path = '/data2/shunliu/pythonfile/code_context_model_prediction/rq1/our/'
    for node_file, edge_file in get_graph_files(dataset_path, mode, step):
        print(node_file)
        nodes = pd.read_csv(node_file)  # columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']
        edges = pd.read_csv(edge_file)  # columns=['start_node_id', 'end_node_id', 'relation', 'label']
        # print('nodes:\n{0} \n edges:\n{1}'.format(nodes, edges))
        src, dst = edges['start_node_id'].tolist(), edges['end_node_id'].tolist()
        g = dgl.graph(data=(src, dst), num_nodes=len(nodes))
        g.ndata['embedding'] = torch.Tensor(nodes['code_embedding'].apply(lambda x: ast.literal_eval(x)).tolist())
        g.ndata['label'] = torch.tensor(nodes['label'].tolist(), dtype=torch.float32)
        kind_mapping = {'variable': 0, 'function': 1, 'class': 2, 'interface': 3}
        nodes['kind_encoded'] = nodes['kind'].map(kind_mapping)
        g.ndata['kind'] = torch.tensor(nodes['kind_encoded'].tolist(), dtype=torch.int64)
        g.edata['relation'] = torch.tensor(edges['relation'].tolist(), dtype=torch.int64)
        # 添加自环边
        g = dgl.add_self_loop(g, edge_feat_names=['relation'], fill_data=5)
        graphs.append(g)
        # print(g, g.nodes(), g.edges())
    return graphs


# 生成包含 20 个图的图数据集
train_graphs = generate_graph_data('train')
print(len(train_graphs))

decoder = Decoder(nembed=1280)
optimizer_de = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=5e-4)

if torch.cuda.is_available():
    decoder = decoder.cuda()


def train(epoch):
    decoder.train()
    total_loss = 0
    total_acc = 0
    for graph in train_graphs:
        optimizer_de.zero_grad()
        features = graph.ndata['embedding'].cuda()  # 节点特征矩阵
        labels = graph.ndata['label'].cuda()  # 节点标签
        idx_train = torch.arange(graph.number_of_nodes()).cuda()
        # 提取边信息
        src, dst = graph.edges()
        num_nodes = graph.number_of_nodes()
        # 手动构建邻接矩阵
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj[src, dst] = 1
        # adj[dst, src] = 1  # 无向图
        adj = adj.cuda()
        embed = features

        ori_num = labels.shape[0]
        embed, labels_new, idx_train_new, adj_up = recon_upsample(embed, labels, idx_train, adj=adj.detach().to_dense())
        generated_G = decoder(embed)

        loss_rec = adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())

        adj_new = copy.deepcopy(generated_G.detach())
        threshold = 0.5
        adj_new[adj_new < threshold] = 0.0
        adj_new[adj_new >= threshold] = 1.0

        # ipdb.set_trace()
        edge_ac = adj_new[:ori_num, :ori_num].eq(adj.to_dense()).double().sum() / (ori_num ** 2)
        total_acc += edge_ac

        # calculate generation information
        # exist_edge_prob = adj_new[:ori_num, :ori_num].mean()  # edge prob for existing nodes
        # generated_edge_prob = adj_new[ori_num:, :ori_num].mean()  # edge prob for generated nodes
        # print("edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(),
        #                                                                                       exist_edge_prob.item(),
        #                                                                                       generated_edge_prob.item()))
        # adj_new = torch.mul(adj_up, adj_new)
        # exist_edge_prob = adj_new[:ori_num, :ori_num].mean()  # edge prob for existing nodes
        # generated_edge_prob = adj_new[ori_num:, :ori_num].mean()  # edge prob for generated nodes
        # print("after filtering, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(
        #     exist_edge_prob.item(), generated_edge_prob.item()))

        total_loss += loss_rec.item()
        loss_rec.backward()
        optimizer_de.step()
    print('Epoch: {:05d}'.format(epoch + 1),
          'loss_rec: {:.4f}'.format(total_loss),
          'acc_rec: {:.4f}'.format(total_acc / len(train_graphs)))


for epoch in range(1000):
    train(epoch)
