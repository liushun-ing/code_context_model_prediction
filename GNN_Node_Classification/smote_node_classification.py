from decimal import Decimal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nni
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAUROC

from .utils_nc import util
from .utils_nc.data_loader import load_prediction_data
from .utils_nc.concat_prediction_model import ConcatPredictionModel
from .utils_nc.attention_prediction_model import AttentionPredictionModel, Encoder, Classifier
from .utils_nc.graph_smote_sampling import Decoder, graph_smote_sampling


def valid(encoder, classifier, data_loader, device, threshold):
    """
    使用验证集验证模型

    :param gnn_model: 模型
    :param data_loader: 图数据加载器
    :param device: device
    :param threshold: classification threshold
    :return: none
    """
    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        criterion = nn.BCELoss()  # 二元交叉熵
        total_loss = 0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        auroc = 0.0
        auprc = 0.0
        f_1 = 0.0
        for g, features, labels, edge_types, kinds in data_loader:
            # output = encoder(g, features, edge_types)
            features_new = encoder(g, features, edge_types)
            output = classifier(g, features_new, edge_types)

            # output = output[torch.eq(seeds, 1)]
            # labels = labels[torch.eq(seeds, 1)]
            # 计算 loss
            loss = criterion(output, labels)
            total_loss += loss.item()
            # print(predict_res, labels)
            # 计算 accuracy
            accuracy_metrics = BinaryAccuracy(threshold=threshold).to(device)
            acc = accuracy_metrics(output, labels).item()
            accuracy += acc
            # 计算 precision
            precision_metrics = BinaryPrecision(threshold=threshold).to(device)
            pre = precision_metrics(output, labels).item()
            precision += pre
            # 计算 recall
            recall_metrics = BinaryRecall(threshold=threshold).to(device)
            rec = recall_metrics(output, labels).item()
            recall += rec
            # 计算F1
            f1_metrics = BinaryF1Score(threshold=threshold).to(device)
            f1 = f1_metrics(output, labels).item()
            f_1 += f1
            # 计算AUC
            metric = BinaryAUROC(thresholds=None).to(device)
            roc = metric(output, labels).item()
            auroc += roc
            # AUPRC
            # metric = BinaryAveragePrecision(thresholds=None).to(device)
            # prc = metric(output, labels.int()).item()
            # auprc += 0 if np.isnan(prc) else prc
            # print(f'precision: {pre}, recall: {rec}, f1_score: {f1}, AUROC: {roc}, AUPRC: {prc}')
        total_loss = total_loss / len(data_loader)
        accuracy = accuracy / len(data_loader)
        precision = precision / len(data_loader)
        recall = recall / len(data_loader)
        auroc = auroc / len(data_loader)
        # auprc = auprc / len(data_loader)
        f_1 = f_1 / len(data_loader)
        print(f'--valid: '
              # f'Loss: {total_loss}, '
              # f'Accuracy: {accuracy}, '
              f'Precision: {Decimal(precision).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")}, '
              f'Recall: {Decimal(recall).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")}, '
              f'F1: {Decimal(f_1).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")}')
        # f'AUROC: {auroc},'
        # f'AUPRC: {auprc}')
        return [total_loss, accuracy, precision, recall, f_1, auroc]


def train(save_path, save_name, step, encoder, decoder, classifier, data_loader, epochs, lr, device, threshold, self_loop,
          load_lazy, weight_decay, smote_weight, use_nni, under_sampling_threshold):
    """
    训练函数

    :param save_path: 保存模型和结果的路径
    :param save_name: 保存结果文件的名字
    :param step: 步长
    :param data_loader: 图数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :param device: 设备 cpu or gpu
    :param threshold: classification threshold
    :param self_loop: whether need self_loop edge
    :param load_lazy: load dataset lazy
    :param weight_decay: adam 权重衰减系数
    :return: none
    """
    encoder.train()
    decoder.train()
    classifier.train()
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()  # 二元交叉熵
    print('----load valid dataset----')
    valid_data_loader = load_prediction_data(save_path, 'valid', batch_size=32, step=step, self_loop=self_loop,
                                             load_lazy=load_lazy, under_sampling_threshold=under_sampling_threshold)
    valid(encoder=encoder, classifier=classifier, data_loader=valid_data_loader, device=device, threshold=threshold)
    result = []
    max_epoch = [0, 0, 0, 0]  # epoch precision recall f1
    best_count = 0
    patience = 20
    for epoch in range(epochs):
        total_classification_loss = 0.0
        total_decoder_loss = 0.0
        train_accuracy = 0.0
        for g, features, labels, edge_types, kinds in data_loader:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_classifier.zero_grad()

            idx_train = torch.arange(g.number_of_nodes()).cuda()
            # 图学习
            embed = encoder(g, features, edge_types)

            # graph smote 生成节点
            g_new, loss_smote = graph_smote_sampling(g, embed, decoder, device)
            features_new = g_new.ndata['embedding']  # 节点特征矩阵
            labels_new = g_new.ndata['label']  # 节点标签
            edge_types_new = g_new.edata['relation']

            # 分类，可以尝试在进行一次图卷积
            output = classifier(g_new, features_new, edge_types_new)
            # output = gnn_model(g, features, edge_types)

            loss = criterion(output, labels_new)
            loss_smote = loss_smote * smote_weight
            total_classification_loss += loss.item()
            total_decoder_loss += loss_smote.item()
            loss = loss + loss_smote
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
            optimizer_classifier.step()

            # 计算 accuracy
            accuracy_metrics = BinaryAccuracy(threshold=threshold).to(device)
            acc = accuracy_metrics(output[idx_train], labels_new[idx_train]).item()
            train_accuracy += acc
        print(f'--train: '
              f'Epoch {epoch}, '
              f'Loss_classification: {Decimal(total_classification_loss / len(data_loader)).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")}, '
              f'Loss_smote: {Decimal(total_decoder_loss / len(data_loader)).quantize(Decimal("0.00001"), rounding="ROUND_HALF_UP")},'
              f'Acc: {Decimal(train_accuracy / len(data_loader)).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")}')
        res = valid(encoder=encoder, classifier=classifier, data_loader=valid_data_loader, device=device, threshold=threshold)
        res.insert(0, epoch)
        res.insert(1, total_classification_loss / len(data_loader))
        res.insert(2, train_accuracy / len(data_loader))
        result.append(res)
        if use_nni:
            nni.report_intermediate_result(res[7])
        # 如何保存最优模型 + 早停机制
        if res[7] > max_epoch[3]:
            # if res[5] + res[6] + res[7] > max_epoch[1] + max_epoch[2] + max_epoch[3]:
            max_epoch = [epoch, res[5], res[6], res[7]]
            best_count = 0
            # 保存最好的模型
            util.save_model(encoder, save_path, step, f'{save_name}_best_encoder')
            util.save_model(classifier, save_path, step, f'{save_name}_best_classifier')
        else:
            best_count += 1
            if best_count >= patience:
                print("Early stopping")
                break
    print(f"best model: {max_epoch}")
    util.save_result(result, save_path, step, save_name)


def start_train(save_path, save_name, step, under_sampling_threshold, encoder, decoder, classifier, device, epochs, lr, batch_size,
                threshold, self_loop, load_lazy, weight_decay, smote_weight, use_nni):
    print('----load train dataset----')
    data_loader = load_prediction_data(save_path, 'train', batch_size, step, under_sampling_threshold, self_loop,
                                       load_lazy)
    # print('----start train----')
    train(save_path, save_name, step, encoder, decoder, classifier, data_loader, epochs, lr, device, threshold, self_loop, load_lazy,
          weight_decay, smote_weight, use_nni, under_sampling_threshold)


def init(model_type, num_layers, in_feats, hidden_size, dropout, num_heads, num_edge_types, use_gpu,
         attention_heads=10, approach='attention'):
    # 定义模型参数  GPU 或 CPU
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    # 创建模型
    if approach == 'concat':
        model = ConcatPredictionModel(model_type, num_layers, in_feats, hidden_size, dropout, num_heads, num_edge_types)
        encoder = Encoder(model_type, num_layers, in_feats, hidden_size, dropout, num_heads,
                          num_edge_types, attention_heads)
        classifier = Classifier(model_type, num_layers, in_feats, hidden_size, dropout, num_heads,
                            num_edge_types, attention_heads)
    elif approach == 'attention':
        encoder = Encoder(model_type, num_layers, in_feats, hidden_size, dropout, num_heads,
                                         num_edge_types, attention_heads)
        classifier = Classifier(model_type, num_layers, in_feats, hidden_size, dropout, num_heads,
                                         num_edge_types, attention_heads)
    else:
        encoder = Encoder(model_type, num_layers, in_feats, hidden_size, dropout, num_heads,
                                         num_edge_types, attention_heads)
        classifier = Classifier(model_type, num_layers, in_feats, hidden_size, dropout, num_heads,
                                         num_edge_types, attention_heads)
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    decoder = Decoder(nembed=hidden_size * num_layers)
    decoder = decoder.to(device)
    return device, encoder, decoder, classifier


def main_func(save_path: str, save_name: str, step: int, under_sampling_threshold=15, model_type="GCN",
              num_layers=3, in_feats=1280, hidden_size=1024, dropout=0.1, attention_heads=8, num_heads=8,
              num_edge_types=6, epochs=50, lr=0.001, batch_size=16, threshold=0.5, use_gpu=True, load_lazy=True,
              weight_decay=1e-6, smote_weight=1e-6, approach='attention', use_nni=False):
    """
    node classification

    :param save_path: 保存模型和结果的路径
    :param save_name: 保存的模型的名字
    :param step: 步长
    :param under_sampling_threshold: 欠采样比例阈值 default: 5.0
    :param model_type: 需要选择的模型： GCN, GAT, RGCN, GraphSAGE, GGNN default:GCN
    :param num_layers: 图卷积层数， default: 3
    :param in_feats: 节点嵌入维度 default: 1280
    :param hidden_size: 图卷积层，隐藏层大小 default: 1024
    :param dropout: dropout rate
    :param attention_heads: number of gnn attention head
    :param num_heads: GAT的注意力机制头数 default: 8
    :param num_edge_types: 边的类型数 default: 6
    :param epochs: 训练次数 default: 50
    :param lr: 学习率 default: 0.001
    :param batch_size: 批次大小 default: 16
    :param threshold: classification threshold
    :param use_gpu: default True
    :param load_lazy: 是否懒加载图数据，default True
    :param weight_decay: Adam 权重衰减系数 default: 1e-6
    :param approach: train approach: attention or concat
    :param use_nni: default False
    :return: None
    """
    print("smote node classification")
    if model_type.startswith('RGCN'):
        self_loop = True
    else:
        self_loop = True
    device, encoder, decoder, classifier = init(model_type, num_layers, in_feats, hidden_size, dropout, num_heads, num_edge_types,
                                  use_gpu,
                                  attention_heads, approach)
    start_train(save_path, save_name, step, under_sampling_threshold, encoder, decoder, classifier, device, epochs, lr, batch_size,
                threshold, self_loop, load_lazy, weight_decay, smote_weight, use_nni)
