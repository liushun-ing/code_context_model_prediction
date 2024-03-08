import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GatedGraphConv


# GAT
class GATModel(nn.Module):
    """
    GATConv based link prediction model: GAT
    """
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=8):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, 1, num_heads=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_types):
        h = self.conv1(g, features)
        # print(h.shape)
        # Concat last 2 dim (num_heads * out_dim)
        h = h.view(-1, h.size(1) * h.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = torch.relu(h)
        h = self.conv2(g, h)
        # Squeeze the head dim as it's = 1
        h = h.squeeze(1)  # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        return torch.sigmoid(h)


# GCN
class GCNModel(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCNModel, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x).squeeze(1)
        return torch.sigmoid(x)


# GraphSAGE
class GraphSAGEModel(nn.Module):
    """
    GraphSAGE layer from Inductive Representation Learning on Large Graphs
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, 1, aggregator_type='mean')

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x).squeeze(1)
        return torch.sigmoid(x)


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
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)
