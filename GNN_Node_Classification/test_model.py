import math
from decimal import Decimal
from os.path import join

import numpy as np
import torch
from dgl import DGLGraph
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryAveragePrecision

from .utils_nc import util
from .utils_nc.data_loader import load_prediction_data
from .utils_nc.prediction_model import GCNModel2, GATModel2, GraphSAGEModel2, GCNModel3, GCNModel4, GATModel3, \
    GATModel4, GraphSAGEModel3, GraphSAGEModel4, GatedGraphModel, RGCNModel3, RGCNModel2, RGCNModel4


def select_result(output: torch.Tensor):
    """如果推荐元素超过十个，就只留 top-75%"""
    indices = (output >= 0.4).nonzero(as_tuple=True)[0].tolist()
    # print(indices)
    if len(indices) < 15:
        return output
    count = math.floor(len(indices) * 0.75)
    for t in range(len(indices), count, -1):
        topk_indices = torch.topk(output, k=t).indices
        # 将第 n 大的元素位置设为 0
        output[topk_indices[-1]] = 0
    return output


def calculate_result(labels: torch.Tensor, output: torch.Tensor, final_k: int, threshold):
    true_number = torch.sum(torch.eq(labels, 1)).item()
    top_k = output[torch.topk(output, k=final_k).indices]
    labels = labels[torch.topk(output, final_k).indices]
    labels = labels[torch.ge(top_k, threshold)]  # top_k 中只选择预测为真的
    true_positive = torch.sum(torch.eq(labels, 1)).item()
    precision = 0 if labels.shape[0] == 0 else true_positive / labels.shape[0]
    recall = true_positive / true_number
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1]


def calculate_result_full(labels, output, threshold, device):
    # 计算 precision
    precision_metrics = BinaryPrecision(threshold=threshold).to(device)
    pre = precision_metrics(output, labels).item()
    # 计算 recall
    recall_metrics = BinaryRecall(threshold=threshold).to(device)
    rec = recall_metrics(output, labels).item()
    # 计算F1
    f1_metrics = BinaryF1Score(threshold=threshold).to(device)
    f1 = f1_metrics(output, labels).item()
    # AUPRC the area under the precision-recall curve
    metric = BinaryAveragePrecision(thresholds=None).to(device)
    prc = metric(output, labels.int()).item()
    prc = 0 if np.isnan(prc) else prc
    return [pre, rec, f1, prc]


def test(gnn_model, data_loader, device, top_k, threshold, fi):
    """
    使用测试集测试最终的模型

    :param gnn_model: 模型
    :param data_loader: 图数据加载器
    :param device: device
    :param top_k: top-k need to prediction 1,3,5, 0-> Full
    :param threshold: classification threshold
    :param fi: file to save result
    :return: none
    """
    with torch.no_grad():
        gnn_model.eval()
        criterion = nn.BCELoss()  # 二元交叉熵
        total_loss = 0.0
        result = []
        for g, features, labels, edge_types in data_loader:
            g = g.to(device)
            features = features.to(device)
            labels = labels.to(device)
            edge_types = edge_types.to(device)

            output = gnn_model(g, features, edge_types)
            # 计算 loss
            loss = criterion(output, labels)
            total_loss += loss.item()
            if top_k != 0:
                final_k = min(len(labels), top_k)
                result.append(calculate_result(labels, output, final_k, threshold))
            else:
                # output = select_result(output)
                # print(labels, output)
                result.append(calculate_result_full(labels, output, threshold, device))
        if top_k != 0:
            p, r, f = 0.0, 0.0, 0.0
            for res in result:
                p += res[0]
                r += res[1]
                f += res[2]
            length = len(result)
            p /= length
            r /= length
            f /= length
            p = Decimal(p).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            r = Decimal(r).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            f = Decimal(f).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            line = f'precision: {p}, recall: {r}, f1_score: {f}\n'
            fi.write(line)
            print(f'{line}')
        else:
            p, r, f, a = 0.0, 0.0, 0.0, 0.0
            for res in result:
                p += res[0]
                r += res[1]
                f += res[2]
                a += res[3]
            length = len(result)
            p /= length
            r /= length
            f /= length
            a /= length
            p = Decimal(p).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            r = Decimal(r).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            f = Decimal(f).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            a = Decimal(a).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            line = f'precision: {p}, recall: {r}, f1_score: {f}, AUPRC: {a}\n'
            fi.write(line)
            print(f'{line}')


def init(model_path, load_name, step, model_name, code_embedding, hidden_size, hidden_size_2, out_feats, use_gpu):
    # 定义模型参数  GPU 或 CPU
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    # 创建模型
    if model_name == 'GAT2':
        print('~~~~~~~~~~~GAT2~~~~~~~~~~~')
        model = GATModel2(code_embedding, hidden_size, out_feats, 0.0, num_heads=8)
    elif model_name == 'GCN2':
        print('~~~~~~~~~~~GCN2~~~~~~~~~~~')
        model = GCNModel2(code_embedding, hidden_size, out_feats, 0.0)
    elif model_name == 'GraphSAGE2':
        print('~~~~~~~~~~~GraphSAGE2~~~~~~~~~~~')
        model = GraphSAGEModel2(code_embedding, hidden_size, out_feats, 0.0)
    elif model_name == 'GAT3':
        print('~~~~~~~~~~~GAT3~~~~~~~~~~~')
        model = GATModel3(code_embedding, hidden_size, out_feats, 0.0, num_heads=8, hidden_size_2=hidden_size_2)
    elif model_name == 'GCN3':
        print('~~~~~~~~~~~GCN3~~~~~~~~~~~')
        model = GCNModel3(code_embedding, hidden_size, out_feats, 0.0, hidden_size_2=hidden_size_2)
    elif model_name == 'GraphSAGE3':
        print('~~~~~~~~~~~GraphSAGE3~~~~~~~~~~~')
        model = GraphSAGEModel3(code_embedding, hidden_size, out_feats, 0.0, hidden_size_2=hidden_size_2)
    elif model_name == 'GAT4':
        print('~~~~~~~~~~~GAT4~~~~~~~~~~~')
        model = GATModel4(code_embedding, hidden_size, out_feats, 0.0, num_heads=8)
    elif model_name == 'GCN4':
        print('~~~~~~~~~~~GCN4~~~~~~~~~~~')
        model = GCNModel4(code_embedding, hidden_size, out_feats, 0.0)
    elif model_name == 'GraphSAGE4':
        print('~~~~~~~~~~~GraphSAGE4~~~~~~~~~~~')
        model = GraphSAGEModel4(code_embedding, hidden_size, out_feats, 0.0)
    elif model_name == 'RGCN2':
        print('~~~~~~~~~~~RGCN2~~~~~~~~~~~')
        model = RGCNModel2(code_embedding, hidden_size, out_feats, 0.0)
    elif model_name == 'RGCN3':
        print('~~~~~~~~~~~RGCN3~~~~~~~~~~~')
        model = RGCNModel3(code_embedding, hidden_size, out_feats, 0.0, hidden_size_2=hidden_size_2)
    elif model_name == 'RGCN4':
        print('~~~~~~~~~~~RGCN4~~~~~~~~~~~')
        model = RGCNModel4(code_embedding, hidden_size, out_feats, 0.0)
    else:
        print('~~~~~~~~~~~else GGNN~~~~~~~~~~~')
        model = GatedGraphModel(code_embedding, hidden_size, out_feats, 5)
    model = util.load_model(model, model_path, step, load_name)
    model.to(device)
    return model, device


def main_func(model_path, load_name, step, model_name='GCN', code_embedding=200, hidden_size=64, hidden_size_2=128,
              out_feats=16,
              use_gpu=True, load_lazy=True):
    """
    测试模型

    :param model_path: path to model
    :param load_name: model name
    :param step: step
    :param model_name: train model name:GCN, GAT, GraphSAGE
    :param code_embedding: the size of code embedding
    :param hidden_size: hidden size of GNN
    :param hidden_size_2: hidden size of second layer
    :param out_feats: out feature size of GNN
    :param use_gpu: default true
    :param load_lazy: load dataset lazy
    :return: None
    """
    model, device = init(model_path, load_name, step, model_name, code_embedding, hidden_size, hidden_size_2, out_feats,
                         use_gpu)
    print('----load test dataset----')
    if model_name.startswith('RGCN'):
        self_loop = False
    else:
        self_loop = True
    data_loader = load_prediction_data(model_path, 'test', batch_size=1, step=step, self_loop=self_loop,
                                       load_lazy=load_lazy)
    print(f'total test graph: {len(data_loader)}')
    # thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = [0.4, 0.5]
    with open(join(model_path, 'result4.txt'), 'a') as f:
        f.write(f'model: {model_name} + step: {step}\n')
        for t in thresholds:
            print()
            for k in [1, 0]:
                # for k in [1, 3, 5, 0]:
                print(f'---threshold:{t} top-k:{k}---')
                f.write(f'---threshold:{t} top-k:{k}---\n')
                test(model, data_loader, device, k, t, f)
        f.close()
