import math
from decimal import Decimal
from os.path import join

import numpy as np
import torch
import nni
from torch import nn
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAveragePrecision

from .utils_nc import util
from .utils_nc.data_loader import load_prediction_data

from .utils_nc.concat_prediction_model import ConcatPredictionModel
from .utils_nc.attention_prediction_model import AttentionPredictionModel


def calculate_result(labels: torch.Tensor, output: torch.Tensor, final_k: int, threshold):
    true_number = torch.sum(torch.eq(labels, 1)).item()
    top_k = output[torch.topk(output, k=final_k).indices]
    labels = labels[torch.topk(output, final_k).indices]
    labels = labels[torch.ge(top_k, threshold)]  # top_k 中只选择预测为真的
    true_positive = torch.sum(torch.eq(labels, 1)).item()
    precision = 0 if labels.shape[0] == 0 else true_positive / labels.shape[0]
    recall = 0 if true_number == 0 else true_positive / true_number
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1]


def calculate_result_full(labels, output, threshold, device):
    res = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if labels.shape[0] == 0:
            print('0')
            res.append([0, 0, 0, 0])
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
        res.append([pre, rec, f1, prc])
    return res


def save_specific_result(labels, output, threshold, kinds, s_file):
    """
    Save specific results based on conditions to a file.

    Parameters:
    labels (torch.Tensor): The labels' tensor.
    output (torch.Tensor): The output tensor.
    threshold (float): The threshold value.
    kinds (torch.Tensor): The kinds' tensor.
    s_file (file object): The file object to write the results to.
    """
    s_file.write('---new predict---\n')
    # Iterate through the tensors and apply the conditions
    for i in range(len(labels)):
        label = labels[i].item()
        out = output[i].item()
        kind = kinds[i].item()
        kind_mapping = ['variable', 'function', 'class', 'interface']
        s_file.write(f"{i} {label} {out} {kind_mapping[int(kind)]} {label == 1}\n")


def print_result(result, threshold):
    target = [0.0, 0.0, 0.0, 0.0]
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'AUPRC':>10}")
    for minConf in s:
        i = s.index(minConf)
        p, r, f, a = 0.0, 0.0, 0.0, 0.0
        for res in result:
            p += res[i][0]
            r += res[i][1]
            f += res[i][2]
            a += res[i][3]
        p = Decimal(p / len(result)).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
        r = Decimal(r / len(result)).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
        f = Decimal(f / len(result)).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
        a = Decimal(a / len(result)).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP")
        print(f"{minConf:>10.1f} {p:>10.3f} {r:>10.3f} {f:>10.3f} {a:>10.3f}")
        if minConf == threshold:
            target = [p, r, f, a]
    return target


def test(gnn_model, data_loader, device, top_k, threshold, use_nni, fi, s_file=None):
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
        for g, features, labels, edge_types, seeds, kinds in data_loader:
            output = gnn_model(g, features, edge_types)
            # output = output[torch.eq(seeds, 0)]
            # labels = labels[torch.eq(seeds, 0)]
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
                if s_file is not None:
                    save_specific_result(labels, output, threshold, kinds, s_file)
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
            target = print_result(result, threshold)
            if use_nni:
                nni.report_final_result(target[2])
            line = f'precision: {target[0]}, recall: {target[1]}, f1_score: {target[2]}, AUPRC: {target[3]}\n'
            fi.write(line)


def init(model_path, load_name, step, model_type, num_layers, in_feats, hidden_size, num_heads, num_edge_types,
         use_gpu, attention_heads=10, approach='attention'):
    # 定义模型参数  GPU 或 CPU
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    # 创建模型
    if approach == 'concat':
        model = ConcatPredictionModel(model_type, num_layers, in_feats, hidden_size, 0, num_heads, num_edge_types)
    elif approach == 'attention':
        model = AttentionPredictionModel(model_type, num_layers, in_feats, hidden_size, 0, num_heads, num_edge_types,
                                         attention_heads)
    else:
        model = AttentionPredictionModel(model_type, num_layers, in_feats, hidden_size, 0, num_heads, num_edge_types,
                                         attention_heads)
    model = util.load_model(model, model_path, step, load_name)
    model.to(device)
    return model, device


def main_func(model_path, load_name, step, model_type="GCN", num_layers=3, in_feats=1280, hidden_size=1024,
              attention_heads=8, num_heads=8, num_edge_types=6, use_gpu=True, load_lazy=True, approach="attention",
              use_nni=False, under_sampling_threshold=15):
    """
    测试模型

    :param model_path: path to trained model
    :param load_name: best model's name
    :param step: step
    :param model_type: train model type: GCN, GAT, GraphSAGE, RGCN, GGNN
    :param num_layers: number of graph convolution layers
    :param in_feats: the size of code embedding
    :param hidden_size: hidden size of GNN
    :param attention_heads: number of graph attention heads
    :param num_heads: number of graph convolution layer attention head
    :param num_edge_types: number of edge type
    :param use_gpu: default true
    :param load_lazy: load dataset lazy
    :param approach: train approach: attention or concat
    :param use_nni: default true
    :param under_sampling_threshold: under sampling threshold
    :return: None
    """
    model, device = init(model_path, load_name, step, model_type, num_layers, in_feats, hidden_size,
                         num_heads, num_edge_types, use_gpu, attention_heads, approach)
    print('----load test dataset----')
    if model_type.startswith('RGCN'):
        self_loop = True
    else:
        self_loop = True
    data_loader = load_prediction_data(model_path, 'test', batch_size=32, step=step, self_loop=self_loop,
                                       load_lazy=load_lazy, under_sampling_threshold=under_sampling_threshold)
    # thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold = 0.4
    with open(join(model_path, 'result4.txt'), 'a') as f:
        f.write(f'model: {model_type} + step: {step}\n')
        for k in [0]:
            # for k in [1, 3, 5, 0]:
            print(f'---top-k:{k}---')
            f.write(f'---top-k:{k}---\n')
            with open(f'specific_result_{step}.txt', 'w') as s_file:
                test(model, data_loader, device, k, threshold, use_nni, f, s_file)
        f.close()
