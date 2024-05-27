import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed)) # 权重矩阵
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))
        return adj_out


def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    """
    使用 smote 进行节点插值生成节点
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
            c_portion = int(avg_number / chosen.shape[0]) # 如果没有设置就采样达到平均值
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

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
    neg_weight = edge_num / (total_num - edge_num)
    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight
    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)
    return loss
