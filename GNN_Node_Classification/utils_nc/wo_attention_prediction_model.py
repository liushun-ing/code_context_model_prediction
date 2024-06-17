import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GatedGraphConv, RelGraphConv


class WoAttentionPredictionModel(nn.Module):
    """
    Concatenate prediction model
    """

    def __init__(self, model_type, number_layers, in_feats, hidden_size, dropout, num_heads, num_edge_types):
        """
        :param model_type: model type [GCN, GAT, GraphSAGE, RGCN, GGNN]
        :param number_layers: graph network neural layers
        :param in_feats: number of input
        :param hidden_size: hidden size
        :param dropout: dropout rate
        :param num_heads: number of attention heads
        :param num_edge_types: number of edge
        """
        super(WoAttentionPredictionModel, self).__init__()
        print("concat prediction model")
        self.model_type = model_type
        self.number_layers = number_layers
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.merge = torch.nn.Linear(in_feats, hidden_size)
        self.gnn_layers = nn.ModuleList()
        self.output = []
        self.mlp = nn.ModuleList()
        if model_type == 'GCN':
            for i in range(self.number_layers):
                self.gnn_layers.append(GraphConv(hidden_size, hidden_size))
        elif model_type == 'GAT':
            self.gnn_layers.append(GATConv(hidden_size, hidden_size, num_heads=num_heads))
            for i in range(self.number_layers - 2):
                self.gnn_layers.append(GATConv(hidden_size * num_heads, hidden_size, num_heads=num_heads))
            self.gnn_layers.append(GATConv(hidden_size * num_heads, hidden_size, num_heads=1))
        elif model_type == 'GraphSAGE':
            for i in range(self.number_layers):
                self.gnn_layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type='mean'))
        elif model_type == 'RGCN':
            for i in range(self.number_layers):
                self.gnn_layers.append(RelGraphConv(hidden_size, hidden_size, num_edge_types, regularizer='basis',
                                                    num_bases=num_edge_types, self_loop=False))
        elif model_type == 'GGNN':
            for i in range(self.number_layers):
                self.gnn_layers.append(GatedGraphConv(hidden_size, hidden_size, n_steps=2, n_etypes=num_edge_types))
        self.dropout = torch.nn.Dropout(p=dropout)
        for i in range(self.number_layers - 1):
            n = self.number_layers - i
            self.mlp.append(torch.nn.Linear(hidden_size * n, hidden_size * (n - 1)))
        self.mlp.append(torch.nn.Linear(hidden_size, 1))

    def forward(self, g, x, edge_types):
        self.output = []
        x = self.merge(x)
        if self.model_type == 'GGNN' or self.model_type == 'RGCN':
            for i in range(self.number_layers):
                x = torch.relu(self.gnn_layers[i](g, x, edge_types))
                if not i == self.number_layers - 1:
                    self.output[i] = x.clone()
                    x = self.dropout(x)
        elif self.model_type == 'GAT':
            for i in range(self.number_layers):
                x = torch.relu(self.gnn_layers[i](g, x)).view(x.shape[0], -1)
                if not i == self.number_layers - 1:
                    self.output.append(x.clone())
                    x = self.dropout(x)
        else:
            for i in range(self.number_layers):
                x = torch.relu(self.gnn_layers[i](g, x))
                if i != self.number_layers - 1:
                    self.output.append(x.clone())
                    x = self.dropout(x)
        self.output.append(x)
        x = torch.cat(self.output, dim=1)
        for i in range(self.number_layers - 1):
            x = torch.relu(self.mlp[i](x))
        x = self.mlp[self.number_layers - 1](x).squeeze(1)
        return torch.sigmoid(x)
