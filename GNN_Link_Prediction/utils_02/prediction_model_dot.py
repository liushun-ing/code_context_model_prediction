import dgl
import torch
from dgl import DGLGraph
import torch.nn as nn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GatedGraphConv


class DotProductPredictor(nn.Module):
    """
    基于点积的边得分评估器
    """
    def forward(self, g: DGLGraph, features):
        """
        根据节点特征，计算节点间点积，作为边存在的可能性

        :param g: 图
        :param features: 当前计算出来的节点特征表示
        :return: 边预测标签
        """
        # 管理一个局部的图上下文。它的主要作用是确保在进入和离开 with 语句块时，
        # 对图进行的修改仅影响到该语句块内的操作，而不会影响到外部的图结构
        with g.local_scope():
            g.ndata['h'] = features
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'predict'))  # 内置函数
            # lambda edges: {'predict': torch.dot(edges.src['h'], edges.dst['h'])}
            return torch.sigmoid(g.edata['predict']).squeeze(1)


# GATConv简单模型
class GATModel(nn.Module):
    """
    GATConv based link prediction model
    """
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=8):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, out_feats, num_heads=1)
        self.pred = DotProductPredictor()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_type):
        # print(features.shape)
        h = self.conv1(g, features)
        # print(h.shape)
        # Concat last 2 dim (num_heads * out_dim)
        h = h.view(-1, h.size(1) * h.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = torch.relu(h)
        h = self.conv2(g, h)
        # Squeeze the head dim as it's = 1
        h = h.squeeze(1)  # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        h = torch.relu(h)
        return self.pred(g, h)


# GCN
class GCNModel(nn.Module):
    """
    GCN: GraphConv based gnn link prediction model
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCNModel, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)
        self.pred = DotProductPredictor()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_type):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        return self.pred(g, x)


# GraphSAGE
class GraphSAGEModel(nn.Module):
    """
    GraphSAGE layer from Inductive Representation Learning on Large Graphs
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, out_feats, aggregator_type='mean')
        self.pred = DotProductPredictor()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


# GGNN
class GatedGraphModel(nn.Module):
    """
    GGNN: Gated Graph Convolution layer from Gated Graph Sequence Neural Networks
    """
    def __init__(self, in_feats, hidden_size, out_feats, num_edge_types):
        super(GatedGraphModel, self).__init__()
        self.conv1 = GatedGraphConv(in_feats, in_feats, n_steps=2, n_etypes=num_edge_types)
        self.conv2 = GatedGraphConv(in_feats, in_feats, n_steps=2, n_etypes=num_edge_types)
        self.conv3 = torch.nn.Linear(in_feats, hidden_size)
        self.conv4 = torch.nn.Linear(hidden_size, out_feats)
        self.pred = DotProductPredictor()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features, edge_types)
        x = torch.relu(x)
        x = self.conv2(g, x, edge_types)
        x = torch.relu(x)
        # 这里可以直接计算点积，也可以先降维后在计算
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)
