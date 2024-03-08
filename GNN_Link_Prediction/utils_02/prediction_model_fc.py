import torch
from dgl import DGLGraph
import torch.nn as nn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GatedGraphConv


def connect_node_embedding(g: DGLGraph, features):
    """
    根据节点特征，拼接边两端节点的特征作为边的特征，传递给后面的全连接层，进行分类任务

    :param g: 图
    :param features: 当前计算出来的节点特征表示
    :return: 边的特征 shape(number of edges, 2 * out_feats)
    """
    # 管理一个局部的图上下文。它的主要作用是确保在进入和离开 with 语句块时，
    # 对图进行的修改仅影响到该语句块内的操作，而不会影响到外部的图结构
    with g.local_scope():
        g.ndata['h'] = features
        g.apply_edges(lambda edges: {'predict': torch.cat([edges.src['h'], edges.dst['h']], dim=1)})  # 内置函数
        # lambda edges: {'predict': torch.dot(edges.src['h'], edges.dst['h'])}
        return g.edata['predict']


# GATConv简单模型
class GATModel(nn.Module):
    """
    GATConv based link prediction model
    """
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=8):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, out_feats, num_heads=1)
        self.pred = torch.nn.Linear(out_feats * 2, 1)  # 拼接后，使用全连接层做预测

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_type):
        # print(features.shape)
        x = self.conv1(g, features)
        # print(h.shape)
        # Concat last 2 dim (num_heads * out_dim)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.conv2(g, x)
        # Squeeze the head dim as it's = 1
        x = x.squeeze(1)  # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        x = torch.relu(x)
        e = connect_node_embedding(g, x)
        e = self.pred(e).squeeze(1)
        return torch.sigmoid(e)


# GCN
class GCNModel(nn.Module):
    """
    GraphConv based gnn link prediction model
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCNModel, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)
        self.pred = torch.nn.Linear(out_feats * 2, 1)
        self.dropout = torch.nn.Dropout(p=0.2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_type):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        g.ndata['h'] = x
        # Use dgl.edge_softmax to calculate edge attention weights
        g.apply_edges(lambda edges: {'score': torch.cat([edges.src['h'], edges.dst['h']], dim=1)})
        return torch.sigmoid(self.pred(g.edata['score']).squeeze())
        # e = connect_node_embedding(g, x)
        # e = self.pred(e).squeeze(1)
        # return torch.sigmoid(e)


# GraphSAGE
class GraphSAGEModel(nn.Module):
    """
    GraphSAGE layer from Inductive Representation Learning on Large Graphs
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, out_feats, aggregator_type='mean')
        self.pred = torch.nn.Linear(out_feats * 2, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        e = connect_node_embedding(g, x)
        e = self.pred(e).squeeze(1)
        return torch.sigmoid(e)


# GGNN
class GatedGraphModel(nn.Module):
    """
    Gated Graph Convolution layer from Gated Graph Sequence Neural Networks
    """
    def __init__(self, in_feats, hidden_size, out_feats, num_edge_types):
        super(GatedGraphModel, self).__init__()
        self.conv1 = GatedGraphConv(in_feats, in_feats, n_steps=2, n_etypes=num_edge_types)
        self.conv2 = GatedGraphConv(in_feats, in_feats, n_steps=2, n_etypes=num_edge_types)
        self.conv3 = torch.nn.Linear(in_feats, hidden_size)
        self.conv4 = torch.nn.Linear(hidden_size, out_feats)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features, edge_types)
        x = torch.relu(x)
        x = self.conv2(g, x, edge_types)
        x = torch.relu(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        e = connect_node_embedding(g, x)
        e = self.pred(e).squeeze(1)
        return torch.sigmoid(e)
