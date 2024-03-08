import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAUROC

from GNN_Link_Prediction.utils_02 import util
from GNN_Link_Prediction.utils_02.data_loader import load_prediction_data
from GNN_Link_Prediction.utils_02.prediction_model_fc import GATModel, GCNModel, GraphSAGEModel, GatedGraphModel
# from GNN_Link_Prediction.utils_02.prediction_model_dot import GCNModel, GATModel, GraphSAGEModel, GatedGraphModel

gnn_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'GNN_Link_Prediction')


def valid(gnn_model, data_loader, device):
    """
    使用验证集验证模型

    :param gnn_model: 模型
    :param data_loader: 图数据加载器
    :param device: device
    :return: none
    """
    with torch.no_grad():
        gnn_model.eval()
        criterion = nn.BCELoss()  # 二元交叉熵
        total_loss = 0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f_1 = 0.0
        auroc = 0.0
        for g, features, labels, edge_types in data_loader:
            g = g.to(device)
            features = features.to(device)
            labels = labels.to(device)
            edge_types = edge_types.to(device)

            output = gnn_model(g, features, edge_types)
            # 计算 loss
            loss = criterion(output, labels)
            total_loss += loss.item()
            # print(predict_res, labels)
            # 计算 accuracy
            accuracy_metrics = BinaryAccuracy().to(device)
            acc = accuracy_metrics(output, labels).item()
            accuracy += acc
            # 计算 precision
            precision_metrics = BinaryPrecision().to(device)
            pre = precision_metrics(output, labels).item()
            precision += pre
            # 计算 recall
            recall_metrics = BinaryRecall().to(device)
            rec = recall_metrics(output, labels).item()
            recall += rec
            # 计算F1
            f1_metrics = BinaryF1Score().to(device)
            f1 = f1_metrics(output, labels).item()
            f_1 += f1
            # 计算AUC
            metric = BinaryAUROC(thresholds=None).to(device)
            auc = metric(output, labels).item()
            auroc += auc
            # print(f'precision: {pre}, recall: {rec}, f1_score: {f1}, AUROC: {auc}')
        print(f'----------valid average result-------\n'
              f'Loss: {total_loss / len(data_loader)}, '
              f'Accuracy: {accuracy / len(data_loader)}, '
              f'Precision: {precision / len(data_loader)}, '
              f'Recall: {recall / len(data_loader)}, '
              f'F1: {f_1 / len(data_loader)}, '
              f'AUROC: {auroc / len(data_loader)}')
        return [total_loss / len(data_loader), accuracy / len(data_loader), precision / len(data_loader),
                recall / len(data_loader), f_1 / len(data_loader), auroc / len(data_loader)]


def train(gnn_model, data_loader, epochs, lr, device, batch_size):
    """
    训练函数

    :param gnn_model: gnn预测
    :param data_loader: 图数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :param device: 设备 cpu or gpu
    :param batch_size: batch大小
    :return: none
    """
    gnn_model.train()
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()  # 二元交叉熵
    print('----load valid dataset----')
    valid_data_loader = load_prediction_data('valid', batch_size=1)  # 加载验证集合
    print(f'valid total graphs: {len(valid_data_loader)}')

    valid(gnn_model=model, data_loader=valid_data_loader, device=device)
    result = []
    for epoch in range(epochs):
        total_loss = 0.0
        train_accuracy = 0.0
        for g, features, labels, edge_types in data_loader:
            g = g.to(device)
            features = features.to(device)
            labels = labels.to(device)
            edge_types = edge_types.to(device)

            optimizer.zero_grad()
            output = gnn_model(g, features, edge_types)
            # print('++++++++++++++++', output, labels)
            # 计算 accuracy
            accuracy_metrics = BinaryAccuracy().to(device)
            acc = accuracy_metrics(output, labels).item()
            train_accuracy += acc
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('------train average result------')
        print(f'Train Epoch {epoch}, '
              f'Average Loss: {total_loss / len(data_loader)}, '
              f'Average Accuracy: {train_accuracy / len(data_loader)}')
        res = valid(gnn_model=model, data_loader=valid_data_loader, device=device)
        res.insert(0, epoch)
        res.insert(1, total_loss / len(data_loader))
        res.insert(2, train_accuracy / len(data_loader))
        result.append(res)
    util.save_result(result, gnn_path, 1)


def start_train(model, device, epochs, lr, batch_size):
    print('----load train dataset----')
    data_loader = load_prediction_data('train', batch_size)
    print(f'train total graphs: {len(data_loader) * batch_size}')
    print('----start train----')
    train(gnn_model=model, data_loader=data_loader, epochs=epochs, lr=lr, device=device, batch_size=batch_size)
    # 保存训练好的模型
    util.save_model(model, gnn_path, 1)


def init():
    # 定义模型参数  GPU 或 CPU
    code_embedding = 200  # 节点特征大小
    hidden_size = 64  # 图卷积层隐藏层大小
    out_feats = 16  # 图卷积层输出层大小（最终线性层的输入，线性层输出为1）
    batch_size = 16  # 每次加载的图数目
    epochs = 30  # 训练轮数r
    lr = 0.001  # 学习率
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 创建模型
    print('~~~~~~~~~~~GAT~~~~~~~~~~~')
    model = GATModel(code_embedding, hidden_size, out_feats, num_heads=8)
    # print('~~~~~~~~~~~GCN~~~~~~~~~~~')
    # model = GCNModel(code_embedding, hidden_size, out_feats)
    # print('~~~~~~~~~~~GraphSAGE~~~~~~~~~~~')
    # model = GraphSAGEModel(code_embedding, hidden_size, out_feats)
    # print('~~~~~~~~~~~GGNN~~~~~~~~~~~')
    # model = GatedGraphModel(code_embedding, hidden_size, out_feats, num_edge_types=5)
    model.to(device)
    model.reset_parameters()
    return model, device, epochs, lr, batch_size


if __name__ == '__main__':
    model, device, epochs, lr, batch_size = init()
    start_train(model, device, epochs, lr, batch_size)
