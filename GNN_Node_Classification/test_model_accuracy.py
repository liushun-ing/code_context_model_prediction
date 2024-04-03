import dgl
from dgl.data import CoraGraphDataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score


dataset = CoraGraphDataset()  # Cora citation network dataset
graph = dataset[0]
graph = dgl.remove_self_loop(graph)  # 消除自环
node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)


class GCNModel3(nn.Module):
    """
    GraphConv based gnn link prediction model: GCN
    """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCNModel3, self).__init__()
        # 不能设置 allow_zero_in_degree,这是需要自行处理，否则没有入度的节点特征将全部变为 0，只能加入自环边
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv3 = GraphConv(hidden_size, out_feats)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pred = torch.nn.Linear(out_feats, n_labels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv3.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, g, features, edge_types):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(g, x)
        x = torch.relu(x)
        x = self.pred(x).squeeze(1)
        return torch.nn.functional.softmax(x, 1)


def valid(gnn_model, device):
    global graph
    global node_labels
    global node_features
    global valid_mask
    with torch.no_grad():
        gnn_model.eval()
        graph = graph.to(device)
        features = node_features.to(device)
        labels = node_labels.to(device)
        valid_mask = valid_mask.to(device)
        output = gnn_model(graph, features, None)[valid_mask]
        labels = labels[valid_mask]
        _, indices = torch.max(output, dim=1)
        correct = torch.sum(indices == labels)
        print(f'valid:----accuracy: {correct.item() * 1.0 / len(labels)}')


def test(gnn_model, device):
    global graph
    global node_labels
    global node_features
    global test_mask
    with torch.no_grad():
        gnn_model.eval()
        graph = graph.to(device)
        features = node_features.to(device)
        labels = node_labels.to(device)
        test_mask = test_mask.to(device)
        output = gnn_model(graph, features, None)[test_mask]
        labels = labels[test_mask]
        _, indices = torch.max(output, dim=1)
        correct = torch.sum(indices == labels)
        print(f'test:----accuracy: {correct.item() * 1.0 / len(labels)}')


def train():
    global graph
    global node_labels
    global node_features
    global train_mask
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gnn_model = GCNModel3(n_features, 512, 128, 0.2).to(device)
    gnn_model.train()
    optimizer = optim.Adam(gnn_model.parameters())
    # optimizer = optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=0.001)
    criterion = torch.nn.functional.cross_entropy  # 二元交叉熵

    for epoch in range(50):
        graph = graph.to(device)
        features = node_features.to(device)
        labels = node_labels.to(device)
        train_mask = train_mask.to(device)

        optimizer.zero_grad()
        output = gnn_model(graph, features, None)[train_mask]
        labels = labels[train_mask]
        # print(output, labels)
        _, indices = torch.max(output, dim=1)
        print(indices, labels)
        correct = torch.sum(indices == labels)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print(f'train:-----accuracy: {correct.item() * 1.0 / len(labels)}, loss: {loss.item()}')
        valid(gnn_model=gnn_model, device=device)
    test(gnn_model=gnn_model, device=device)

train()
