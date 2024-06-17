import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GatedGraphConv, RelGraphConv


def src_mul_edge(src_field, edge_field, out_field):
    def func(edges):
        return {out_field: edges.src[src_field] * edges.data[edge_field]}

    return func


def node_sum(in_field, out_field):
    def func(nodes):
        return {out_field: torch.sum(nodes.mailbox[in_field], dim=1)}

    return func


def copy_edge(src_field, dst_field):
    def func(edges):
        return {dst_field: edges.data[src_field]}

    return func


def edge_sum(in_field, out_field):
    def func(nodes):
        return {out_field: torch.sum(nodes.mailbox[in_field], dim=1)}

    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}
        # return {field: torch.exp((edges.data[field] / scale_constant))}

    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, model_type, hidden_size, dropout, attention_heads, num_heads, num_edge_types):
        super().__init__()
        self.model_type = model_type
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        if model_type == 'GCN':
            self.feature_Q = GraphConv(in_feats=hidden_size, out_feats=hidden_size)
            self.feature_K = GraphConv(in_feats=hidden_size, out_feats=hidden_size)
            self.feature_V = GraphConv(in_feats=hidden_size, out_feats=hidden_size)
        elif model_type == 'GAT':
            self.feature_Q = GATConv(hidden_size, hidden_size, num_heads=num_heads)
            self.feature_K = GATConv(hidden_size, hidden_size, num_heads=num_heads)
            self.feature_V = GATConv(hidden_size, hidden_size, num_heads=num_heads)
        elif model_type == 'GraphSAGE':
            self.feature_Q = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
            self.feature_K = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
            self.feature_V = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
        elif model_type == 'RGCN':
            self.feature_Q = RelGraphConv(hidden_size, hidden_size, num_edge_types, regularizer='basis',
                                          num_bases=num_edge_types, self_loop=False)
            self.feature_K = RelGraphConv(hidden_size, hidden_size, num_edge_types, regularizer='basis',
                                          num_bases=num_edge_types, self_loop=False)
            self.feature_V = RelGraphConv(hidden_size, hidden_size, num_edge_types, regularizer='basis',
                                          num_bases=num_edge_types, self_loop=False)
        elif model_type == 'GGNN':
            self.feature_Q = GatedGraphConv(hidden_size, hidden_size, n_steps=2, n_etypes=num_edge_types)
            self.feature_K = GatedGraphConv(hidden_size, hidden_size, n_steps=2, n_etypes=num_edge_types)
            self.feature_V = GatedGraphConv(hidden_size, hidden_size, n_steps=2, n_etypes=num_edge_types)
        self.dropout = torch.nn.Dropout(p=dropout)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.hidden_size // self.attention_heads)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):
        with g.local_scope():
            if self.model_type == 'GCN' or self.model_type == 'GraphSAGE':
                Q_h = F.relu(self.feature_Q(g, self.dropout(h)))
                K_h = F.relu(self.feature_K(g, self.dropout(h)))
                V_h = F.relu(self.feature_V(g, self.dropout(h)))
            elif self.model_type == 'GAT':
                q = self.feature_Q(g, self.dropout(h))
                Q_h = F.relu(q).view(q.shape[0], -1)
                k = self.feature_Q(g, self.dropout(h))
                K_h = F.relu(q).view(k.shape[0], -1)
                v = self.feature_Q(g, self.dropout(h))
                V_h = F.relu(q).view(v.shape[0], -1)
            elif self.model_type == 'RGCN' or self.model_type == 'GGNN':
                Q_h = F.relu(self.feature_Q(g, self.dropout(h), e))
                K_h = F.relu(self.feature_K(g, self.dropout(h), e))
                V_h = F.relu(self.feature_V(g, self.dropout(h), e))
            # Reshaping into [num_nodes, num_heads, feat_dim] to
            # get projections for multi-head attention
            g.ndata['Q_h'] = Q_h.view(-1, self.attention_heads, self.hidden_size // self.attention_heads)
            g.ndata['K_h'] = K_h.view(-1, self.attention_heads, self.hidden_size // self.attention_heads)
            g.ndata['V_h'] = V_h.view(-1, self.attention_heads, self.hidden_size // self.attention_heads)

            self.propagate_attention(g)

            # head_out = g.ndata['wV'] / g.ndata['z'] 避免出现 0 的情况，但实际上不可能，因为都加了自旋边
            head_out = g.ndata['wV'] / (
                    g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
            return head_out


class WoConcatPredictionModel(nn.Module):
    """
    Attention prediction model
    """

    def __init__(self, model_type, number_layers, in_feats, hidden_size, dropout=0.1, attention_heads=10, num_heads=8,
                 num_edge_types=6):
        """
        :param model_type: model type [GCN, GAT, GraphSAGE, RGCN, GGNN]
        :param number_layers: graph network neural layers
        :param in_feats: number of input
        :param hidden_size: hidden size
        :param dropout: dropout rate, defaults to 0.1
        :param attention_heads: number of graph attention heads, defaults to 10
        :param num_heads: number of gat attention heads, defaults to 8
        :param num_edge_types: number of edge, defaults to 6
        """
        super(WoConcatPredictionModel, self).__init__()
        print("attention prediction model")
        self.number_layers = number_layers
        self.hidden_size = hidden_size
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.merge = torch.nn.Linear(in_feats, hidden_size)
        self.gnn_attention_layers = nn.ModuleList()
        self.mlp = nn.ModuleList()
        for i in range(self.number_layers):
            self.gnn_attention_layers.append(
                MultiHeadAttentionLayer(model_type, hidden_size, dropout, attention_heads, num_heads, num_edge_types))
        for i in range(self.number_layers - 1):
            self.mlp.append(torch.nn.Linear(hidden_size // pow(2, i), hidden_size // pow(2, i + 1)))
        self.mlp.append(torch.nn.Linear(hidden_size // pow(2, self.number_layers - 1), 1))

    def forward(self, g, x, edge_types):
        x = F.relu(self.merge(x))
        for i in range(self.number_layers):
            x = self.gnn_attention_layers[i](g, x, edge_types).view(-1, self.hidden_size)
        for i in range(self.number_layers - 1):
            x = torch.relu(self.mlp[i](x))
        x = self.mlp[self.number_layers - 1](x).squeeze(1)
        return torch.sigmoid(x)
