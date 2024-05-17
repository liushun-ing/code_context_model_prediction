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
    def __init__(self, num_heads, in_feats, out_feats, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.feature_Q = GraphConv(in_feats=in_feats, out_feats=out_feats)
        self.feature_K = GraphConv(in_feats=in_feats, out_feats=out_feats)
        self.feature_V = GraphConv(in_feats=in_feats, out_feats=out_feats)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.feature_Q.reset_parameters()
        self.feature_K.reset_parameters()
        self.feature_V.reset_parameters()

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_feats // self.num_heads)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h):
        with g.local_scope():
            Q_h = F.relu(self.feature_Q(g, self.dropout(h)))  # self.Q(feature_Q)
            K_h = F.relu(self.feature_K(g, self.dropout(h)))  # self.K(feature_K)
            V_h = F.relu(self.feature_V(g, self.dropout(h)))  # self.V(feature_V)

            # Reshaping into [num_nodes, num_heads, feat_dim] to
            # get projections for multi-head attention
            g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_feats // self.num_heads)
            g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_feats // self.num_heads)
            g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_feats // self.num_heads)

            self.propagate_attention(g)

            # head_out = g.ndata['wV'] / g.ndata['z'] 避免出现 0 的情况，但实际上不可能，因为都加了自旋边
            head_out = g.ndata['wV'] / (
                    g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
            return head_out


# class GCNModel3(nn.Module):
#     """
#     GraphConv based gnn link prediction model: GCN
#     """
#     output_1, output_2 = None, None
#
#     def __init__(self, in_feats, hidden_size, out_feats, dropout, hidden_size_2=128, graph_size=512):
#         super(GCNModel3, self).__init__()
#         print("GCN attention prediction model")
#         # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
#         self.graph_size = graph_size
#         self.merge1 = torch.nn.Linear(in_feats, graph_size * 2)
#         self.attention1 = MultiHeadAttentionLayer(4, graph_size * 2, dropout)
#         self.merge2 = torch.nn.Linear(graph_size * 2, graph_size)
#         self.attention2 = MultiHeadAttentionLayer(4, graph_size, dropout)
#         self.mlp1 = torch.nn.Linear(graph_size, graph_size // 2)  # 512 - 256
#         self.mlp2 = torch.nn.Linear(graph_size // 2, graph_size // 8)  # 256 - 64
#         self.mlp3 = torch.nn.Linear(graph_size // 8, 1)  # 64 -1
#
#     def reset_parameters(self):
#         self.merge1.reset_parameters()
#         self.merge2.reset_parameters()
#         self.attention1.reset_parameters()
#         self.attention2.reset_parameters()
#         self.mlp1.reset_parameters()
#         self.mlp2.reset_parameters()
#         self.mlp3.reset_parameters()
#
#     def forward(self, g, features, edge_types):
#         x = self.merge1(features)
#         x = F.relu(x)
#         x = self.attention1(g, x)
#         x = x.view(-1, self.graph_size * 2)
#         x = self.merge2(x)
#         x = F.relu(x)
#         x = self.attention2(g, x)
#         x = x.view(-1, self.graph_size)
#         x = self.mlp1(x)
#         x = F.relu(x)
#         x = self.mlp2(x)
#         x = F.relu(x)
#         x = self.mlp3(x).squeeze(1)
#         return torch.sigmoid(x)

class GCNModel3(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    output_1, output_2 = None, None

    def __init__(self, in_feats, hidden_size, out_feats, dropout, hidden_size_2=128):
        super(GCNModel3, self).__init__()
        print("3 layer GCN attention concat prediction model")
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.out_feats = out_feats
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.attention1 = MultiHeadAttentionLayer(4, in_feats, in_feats, dropout)
        self.merge1 = torch.nn.Linear(in_feats, hidden_size)
        self.attention2 = MultiHeadAttentionLayer(4, hidden_size, hidden_size, dropout)
        self.merge2 = torch.nn.Linear(hidden_size, hidden_size_2)
        self.attention3 = MultiHeadAttentionLayer(4, hidden_size_2, hidden_size_2, dropout)
        self.merge3 = torch.nn.Linear(hidden_size_2, out_feats)
        self.mlp1 = torch.nn.Linear(hidden_size + hidden_size_2 + out_feats, hidden_size_2 + out_feats)
        self.mlp2 = torch.nn.Linear(hidden_size_2 + out_feats, out_feats)
        self.mlp3 = torch.nn.Linear(out_feats, 1)

    def reset_parameters(self):
        self.merge1.reset_parameters()
        self.merge2.reset_parameters()
        self.merge3.reset_parameters()
        self.attention1.reset_parameters()
        self.attention2.reset_parameters()
        self.attention3.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.attention1(g, features)
        x = x.view(-1, self.in_feats)
        x = self.merge1(x)
        x = F.relu(x)
        self.output_1 = x.clone()
        x = self.attention2(g, x)
        x = x.view(-1, self.hidden_size)
        x = self.merge2(x)
        x = F.relu(x)
        self.output_2 = x.clone()
        x = self.attention3(g, x)
        x = x.view(-1, self.hidden_size_2)
        x = self.merge3(x)
        x = F.relu(x)
        x = torch.cat((self.output_1, self.output_2, x), dim=1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)
        x = self.mlp3(x).squeeze(1)
        return torch.sigmoid(x)


class RMultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads, graph_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.graph_size = graph_size
        self.feature_Q = RelGraphConv(in_feat=graph_size, out_feat=graph_size, num_rels=6, regularizer='basis',
                                      num_bases=6, activation=F.relu, dropout=dropout, self_loop=False)
        self.feature_K = RelGraphConv(in_feat=graph_size, out_feat=graph_size, num_rels=6, regularizer='basis',
                                      num_bases=6, activation=F.relu, dropout=dropout, self_loop=False)
        self.feature_V = RelGraphConv(in_feat=graph_size, out_feat=graph_size, num_rels=6, regularizer='basis',
                                      num_bases=6, activation=F.relu, dropout=dropout, self_loop=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.graph_size // self.num_heads)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):
        with g.local_scope():
            Q_h = self.feature_Q(g, h, e)  # self.Q(feature_Q)
            K_h = self.feature_K(g, h, e)  # self.K(feature_K)
            V_h = self.feature_V(g, h, e)  # self.V(feature_V)

            # Reshaping into [num_nodes, num_heads, feat_dim] to
            # get projections for multi-head attention
            g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.graph_size // self.num_heads)
            g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.graph_size // self.num_heads)
            g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.graph_size // self.num_heads)

            self.propagate_attention(g)

            # head_out = g.ndata['wV'] / g.ndata['z'] 避免出现 0 的情况，但实际上不可能，因为都加了自旋边
            head_out = g.ndata['wV'] / (
                    g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
            return head_out


# class RGCNModel3(nn.Module):
#     """
#     GraphConv based gnn link prediction model: GCN
#     """
#     output_1, output_2 = None, None
#
#     def __init__(self, in_feats, hidden_size, out_feats, dropout, hidden_size_2=128, graph_size=512):
#         super(RGCNModel3, self).__init__()
#         print("RGCN attention prediction model")
#         # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
#         self.graph_size = graph_size
#         self.merge1 = torch.nn.Linear(in_feats, graph_size * 2)
#         self.attention1 = RMultiHeadAttentionLayer(4, graph_size * 2, dropout)
#         self.merge2 = torch.nn.Linear(graph_size * 2, graph_size)
#         self.attention2 = RMultiHeadAttentionLayer(4, graph_size, dropout)
#         self.mlp1 = torch.nn.Linear(graph_size, graph_size // 2)  # 512 - 256
#         self.mlp2 = torch.nn.Linear(graph_size // 2, graph_size // 8)  # 256 - 64
#         self.mlp3 = torch.nn.Linear(graph_size // 8, 1)  # 64 -1
#
#     def reset_parameters(self):
#         self.merge1.reset_parameters()
#         self.merge2.reset_parameters()
#         self.mlp1.reset_parameters()
#         self.mlp2.reset_parameters()
#         self.mlp3.reset_parameters()
#
#     def forward(self, g, features, edge_types):
#         x = self.merge1(features)
#         x = F.relu(x)
#         x = self.attention1(g, x, edge_types)
#         x = x.view(-1, self.graph_size * 2)
#         x = self.merge2(x)
#         x = F.relu(x)
#         x = self.attention2(g, x, edge_types)
#         x = x.view(-1, self.graph_size)
#         x = self.mlp1(x)
#         x = F.relu(x)
#         x = self.mlp2(x)
#         x = F.relu(x)
#         x = self.mlp3(x).squeeze(1)
#         return torch.sigmoid(x)

class RGCNModel3(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    output_1, output_2 = None, None

    def __init__(self, in_feats, hidden_size, out_feats, dropout, hidden_size_2=128, graph_size=512):
        super(RGCNModel3, self).__init__()
        print("RGCN attention concat prediction model")
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.graph_size = graph_size
        self.merge1 = torch.nn.Linear(in_feats, graph_size * 2)
        self.attention1 = RMultiHeadAttentionLayer(4, graph_size * 2, dropout)
        self.merge2 = torch.nn.Linear(graph_size * 2, graph_size)
        self.attention2 = RMultiHeadAttentionLayer(4, graph_size, dropout)
        self.mlp1 = torch.nn.Linear(graph_size * 3, graph_size)  # 1536 - 512
        self.mlp2 = torch.nn.Linear(graph_size, graph_size // 4)  # 512 - 128
        self.mlp3 = torch.nn.Linear(graph_size // 4, 1)  # 128 -1

    def reset_parameters(self):
        self.merge1.reset_parameters()
        self.merge2.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.merge1(features)
        x = F.relu(x)
        x = self.attention1(g, x, edge_types)
        x = x.view(-1, self.graph_size * 2)
        self.output_1 = x
        x = self.merge2(x)
        x = F.relu(x)
        x = self.attention2(g, x, edge_types)
        x = x.view(-1, self.graph_size)
        self.output_2 = x
        x = torch.cat((self.output_1, self.output_2), dim=1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)
        x = self.mlp3(x).squeeze(1)
        return torch.sigmoid(x)