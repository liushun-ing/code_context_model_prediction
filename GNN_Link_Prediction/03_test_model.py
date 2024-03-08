import os
from os.path import join

import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAUROC

from GNN_Link_Prediction.utils_02 import util
from GNN_Link_Prediction.utils_02.data_loader import load_prediction_data
from GNN_Link_Prediction.utils_02.prediction_model_fc import GCNModel, GATModel, GraphSAGEModel, GatedGraphModel
# from GNN_Link_Prediction.utils_02.prediction_model_dot import GCNModel, GATModel, GraphSAGEModel, GatedGraphModel


def test(gnn_model, data_loader, device):
    """
    使用测试集测试最终的模型

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
            # labels = labels[torch.topk(output, 1).indices]
            # output = output[torch.topk(output, 1).indices]
            output = output[edge_types != 4]
            labels = labels[edge_types != 4]
            o = util.threshold_tensor(output)
            print(o, labels)
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
            print(f'accuracy: {acc}, precision: {pre}, recall: {rec}, f1_score: {f1}, AUROC: {auc}')
        print(f'----------test average result-------\n'
              f'Loss: {total_loss / len(data_loader)}, '
              f'Accuracy: {accuracy/ len(data_loader)}, '
              f'Precision: {precision / len(data_loader)}, '
              f'Recall: {recall / len(data_loader)}, '
              f'F1: {f_1 / len(data_loader)}, '
              f'AUROC: {auroc / len(data_loader)}')


def init():
    # 定义模型参数  GPU 或 CPU
    code_embedding = 200
    hidden_size = 64
    out_feats = 16
    batch_size = 1
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    gnn_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'GNN_Link_Prediction')
    # 创建模型
    print('~~~~~~~~~~~GAT~~~~~~~~~~~')
    model = GATModel(code_embedding, hidden_size, out_feats, num_heads=8)
    # print('~~~~~~~~~~~GCN~~~~~~~~~~~')
    # model = GCNModel(code_embedding, hidden_size, out_feats)
    # print('~~~~~~~~~~~GraphSAGE~~~~~~~~~~~')
    # model = GraphSAGEModel(code_embedding, hidden_size, out_feats)
    # print('~~~~~~~~~~~GGNN~~~~~~~~~~~')
    # model = GatedGraphModel(code_embedding, hidden_size, out_feats, num_edge_types=5)
    model = util.load_model(model, gnn_path, 1)
    model.to(device)
    return model, device, batch_size


if __name__ == '__main__':
    model, device, batch_size = init()
    print('----load test dataset----')
    data_loader = load_prediction_data('test', batch_size)
    test(model, data_loader, device)
