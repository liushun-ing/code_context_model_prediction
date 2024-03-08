import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GatedGraphConv, RelGraphConv


# GAT
class GATModel2(nn.Module):
    """
    GATConv based link prediction model: GAT
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout, num_heads=8):
        super(GATModel2, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, out_feats, num_heads=1)
        self.pred = torch.nn.Linear(out_feats, 1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        # print(h.shape)
        # Concat last 2 dim (num_heads * out_dim)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        # Squeeze the head dim as it's = 1
        x = x.squeeze(1)  # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)

class GATModel3(nn.Module):
    """
    GATConv based link prediction model: GAT
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout, num_heads=8):
        super(GATModel3, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, hidden_size, num_heads=4)
        self.conv3 = GATConv(hidden_size * 4, out_feats, num_heads=1)
        self.pred = torch.nn.Linear(out_feats, 1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = x.squeeze(1)  # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class GATModel4(nn.Module):
    """
    GATConv based link prediction model: GAT
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout, num_heads=8):
        super(GATModel4, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, hidden_size, num_heads=4)
        self.conv3 = GATConv(hidden_size * 4, hidden_size, num_heads=2)
        self.conv4 = GATConv(hidden_size * 2, out_feats, num_heads=1)
        self.pred = torch.nn.Linear(out_feats, 1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = x.view(-1, x.size(1) * x.size(2))  # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv4(g, x)
        x = x.squeeze(1)  # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


# GCN
class GCNModel2(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCNModel2, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class GCNModel3(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCNModel3, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, out_feats)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class GCNModel4(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCNModel4, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, hidden_size)
        self.conv4 = GraphConv(hidden_size, out_feats)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv4(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class RGCNModel2(nn.Module):
    """
    RelGraphConv based gnn link prediction model: RGCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(RGCNModel2, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = RelGraphConv(in_feats, hidden_size, 4, regularizer='basis', num_bases=2)
        self.conv2 = RelGraphConv(hidden_size, out_feats, 4, regularizer='basis', num_bases=2)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features, edge_types)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x, edge_types)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class RGCNModel3(nn.Module):
    """
    RelGraphConv based gnn link prediction model: RGCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(RGCNModel3, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = RelGraphConv(in_feats, hidden_size, 4, regularizer='basis', num_bases=2)
        self.conv2 = RelGraphConv(hidden_size, hidden_size, 4, regularizer='basis', num_bases=2)
        self.conv3 = RelGraphConv(hidden_size, out_feats, 4, regularizer='basis', num_bases=2)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features, edge_types)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x, edge_types)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x, edge_types)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class RGCNModel4(nn.Module):
    """
    RelGraphConv based gnn link prediction model: RGCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(RGCNModel4, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = RelGraphConv(in_feats, hidden_size, 4, regularizer='basis', num_bases=2)
        self.conv2 = RelGraphConv(hidden_size, hidden_size, 4, regularizer='basis', num_bases=2)
        self.conv3 = RelGraphConv(hidden_size, hidden_size, 4, regularizer='basis', num_bases=2)
        self.conv4 = RelGraphConv(hidden_size, out_feats, 4, regularizer='basis', num_bases=2)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features, edge_types)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x, edge_types)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x, edge_types)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv4(g, x, edge_types)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


# GraphSAGE
class GraphSAGEModel2(nn.Module):
    """
    GraphSAGE layer from Inductive Representation Learning on Large Graphs
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GraphSAGEModel2, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, out_feats, aggregator_type='mean')
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class GraphSAGEModel3(nn.Module):
    """
    GraphSAGE layer from Inductive Representation Learning on Large Graphs
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GraphSAGEModel3, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
        self.conv3 = SAGEConv(hidden_size, out_feats, aggregator_type='mean')
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.sigmoid(x)


class GraphSAGEModel4(nn.Module):
    """
    GraphSAGE layer from Inductive Representation Learning on Large Graphs
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GraphSAGEModel4, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
        self.conv3 = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
        self.conv4 = SAGEConv(hidden_size, out_feats, aggregator_type='mean')
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv4(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
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
