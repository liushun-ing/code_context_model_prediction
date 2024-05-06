import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAUROC

from .utils_nc import util
from .utils_nc.data_loader import load_prediction_data
# from .utils_nc.prediction_model import GATModel2, GCNModel2, GraphSAGEModel2, GATModel3, GATModel4, GCNModel3, \
#     GCNModel4, GraphSAGEModel3, GraphSAGEModel4, GatedGraphModel, RGCNModel3, RGCNModel4, RGCNModel2


from .utils_nc.concat_prediction_model import GATModel2, GCNModel2, GraphSAGEModel2, GATModel3, GATModel4, GCNModel3, \
    GCNModel4, GraphSAGEModel3, GraphSAGEModel4, GatedGraphModel, RGCNModel3, RGCNModel4, RGCNModel2

# from .utils_nc.merge_prediction_model import GATModel2, GCNModel2, GraphSAGEModel2, GATModel3, GATModel4, GCNModel3, \
#     GCNModel4, GraphSAGEModel3, GraphSAGEModel4, GatedGraphModel, RGCNModel3, RGCNModel4, RGCNModel2

def valid(gnn_model, data_loader, device, threshold):
    """
    使用验证集验证模型

    :param gnn_model: 模型
    :param data_loader: 图数据加载器
    :param device: device
    :param threshold: classification threshold
    :return: none
    """
    with torch.no_grad():
        gnn_model.eval()
        criterion = nn.BCELoss()  # 二元交叉熵
        total_loss = 0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        auroc = 0.0
        auprc = 0.0
        f_1 = 0.0
        for g, features, labels, edge_types, seeds in data_loader:
            g = g.to(device)
            features = features.to(device)
            labels = labels.to(device)
            edge_types = edge_types.to(device)
            seeds = seeds.to(device)

            output = gnn_model(g, features, edge_types)
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
            metric = BinaryAveragePrecision(thresholds=None).to(device)
            prc = metric(output, labels.int()).item()
            auprc += 0 if np.isnan(prc) else prc
            # print(f'precision: {pre}, recall: {rec}, f1_score: {f1}, AUROC: {roc}, AUPRC: {prc}')
        total_loss = total_loss / len(data_loader)
        accuracy = accuracy / len(data_loader)
        precision = precision / len(data_loader)
        recall = recall / len(data_loader)
        auroc = auroc / len(data_loader)
        auprc = auprc / len(data_loader)
        f_1 = f_1 / len(data_loader)
        # if precision + recall == 0:
        #     f_1 = 0
        # else:
        #     f_1 = 2 * precision * recall / (precision + recall)  # 使用precision和recall的平均值计算
        print(f'----------valid average result-------\n'
              f'Loss: {total_loss}, '
              f'Accuracy: {accuracy}, '
              f'Precision: {precision}, '
              f'Recall: {recall}, '
              f'F1: {f_1}, '
              f'AUROC: {auroc},'
              f'AUPRC: {auprc}')
        return [total_loss, accuracy, precision, recall, f_1, auroc]


def train(save_path, save_name, step, gnn_model, data_loader, epochs, lr, device, threshold, self_loop, load_lazy):
    """
    训练函数

    :param save_path: 保存模型和结果的路径
    :param save_name: 保存结果文件的名字
    :param step: 步长
    :param gnn_model: gnn预测
    :param data_loader: 图数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :param device: 设备 cpu or gpu
    :param threshold: classification threshold
    :param self_loop: whether need self_loop edge
    :param load_lazy: load dataset lazy
    :return: none
    """
    gnn_model.train()
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
    # optimizer = optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.BCELoss()  # 二元交叉熵
    print('----load valid dataset----')
    valid_data_loader = load_prediction_data(save_path, 'valid', batch_size=1, step=step, self_loop=self_loop,
                                             load_lazy=load_lazy)  # 加载验证集合
    print(f'valid total graphs: {len(valid_data_loader)}')
    valid(gnn_model=gnn_model, data_loader=valid_data_loader, device=device, threshold=threshold)
    result = []
    max_epoch = [0, 0, 0, 0]  # precision recall f1
    for epoch in range(epochs):
        total_loss = 0.0
        train_accuracy = 0.0
        for g, features, labels, edge_types, seeds in data_loader:
            g = g.to(device)
            features = features.to(device)
            labels = labels.to(device)
            edge_types = edge_types.to(device)
            seeds = seeds.to(device)

            optimizer.zero_grad()
            output = gnn_model(g, features, edge_types)
            # print('++++++++++++++++', output, labels)
            # 计算 accuracy
            accuracy_metrics = BinaryAccuracy(threshold=threshold).to(device)
            # 将 seed 为1的节点丢弃
            # output = output[torch.eq(seeds, 0)]
            # labels = labels[torch.eq(seeds, 0)]
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
        res = valid(gnn_model=gnn_model, data_loader=valid_data_loader, device=device, threshold=threshold)
        res.insert(0, epoch)
        res.insert(1, total_loss / len(data_loader))
        res.insert(2, train_accuracy / len(data_loader))
        result.append(res)
        # 如何保存最优模型
        if res[7] > max_epoch[3]:
        # if res[5] + res[6] + res[7] > max_epoch[1] + max_epoch[2] + max_epoch[3]:
            max_epoch = [epoch, res[5], res[6], res[7]]
        # 保存训练好的模型
        util.save_model(gnn_model, save_path, step, f'{save_name}_{epoch}')
    util.save_result(result, save_path, step, save_name)
    # 保留最好的模型，其余的模型丢弃
    print('the best model is: ', max_epoch)
    util.maintain_best_model(save_path, step, save_name, max_epoch[0])


def start_train(save_path, save_name, step, under_sampling_threshold, model, device, epochs, lr, batch_size, threshold,
                self_loop, load_lazy):
    print('----load train dataset----')
    data_loader = load_prediction_data(save_path, 'train', batch_size, step, under_sampling_threshold, self_loop,
                                       load_lazy)
    print(f'train total graphs: {len(data_loader) * batch_size}')
    print('----start train----')
    train(save_path, save_name, step, model, data_loader, epochs, lr, device, threshold, self_loop, load_lazy)


def init(model_name, code_embedding, hidden_size, hidden_size_2, out_feats, dropout, use_gpu):
    # 定义模型参数  GPU 或 CPU
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    # 创建模型
    if model_name == 'GAT2':
        print('~~~~~~~~~~~GAT2~~~~~~~~~~~')
        model = GATModel2(code_embedding, hidden_size, out_feats, dropout, num_heads=8)
    elif model_name == 'GCN2':
        print('~~~~~~~~~~~GCN2~~~~~~~~~~~')
        model = GCNModel2(code_embedding, hidden_size, out_feats, dropout)
    elif model_name == 'GraphSAGE2':
        print('~~~~~~~~~~~GraphSAGE2~~~~~~~~~~~')
        model = GraphSAGEModel2(code_embedding, hidden_size, out_feats, dropout)
    elif model_name == 'GAT3':
        print('~~~~~~~~~~~GAT3~~~~~~~~~~~')
        model = GATModel3(code_embedding, hidden_size, out_feats, dropout, num_heads=8, hidden_size_2=hidden_size_2)
    elif model_name == 'GCN3':
        print('~~~~~~~~~~~GCN3~~~~~~~~~~~')
        model = GCNModel3(code_embedding, hidden_size, out_feats, dropout, hidden_size_2=hidden_size_2)
    elif model_name == 'GraphSAGE3':
        print('~~~~~~~~~~~GraphSAGE3~~~~~~~~~~~')
        model = GraphSAGEModel3(code_embedding, hidden_size, out_feats, dropout, hidden_size_2=hidden_size_2)
    elif model_name == 'GAT4':
        print('~~~~~~~~~~~GAT4~~~~~~~~~~~')
        model = GATModel4(code_embedding, hidden_size, out_feats, dropout, num_heads=8)
    elif model_name == 'GCN4':
        print('~~~~~~~~~~~GCN4~~~~~~~~~~~')
        model = GCNModel4(code_embedding, hidden_size, out_feats, dropout, hidden_size_2=hidden_size_2)
    elif model_name == 'GraphSAGE4':
        print('~~~~~~~~~~~GraphSAGE4~~~~~~~~~~~')
        model = GraphSAGEModel4(code_embedding, hidden_size, out_feats, dropout)
    elif model_name == 'RGCN2':
        print('~~~~~~~~~~~RGCN2~~~~~~~~~~~')
        model = RGCNModel2(code_embedding, hidden_size, out_feats, dropout)
    elif model_name == 'RGCN3':
        print('~~~~~~~~~~~RGCN3~~~~~~~~~~~')
        model = RGCNModel3(code_embedding, hidden_size, out_feats, dropout, hidden_size_2=hidden_size_2)
    elif model_name == 'RGCN4':
        print('~~~~~~~~~~~RGCN4~~~~~~~~~~~')
        model = RGCNModel4(code_embedding, hidden_size, out_feats, dropout)
    else:
        print('~~~~~~~~~~~default GGNN~~~~~~~~~~~')
        model = GatedGraphModel(code_embedding, hidden_size, out_feats, 5)
    model.to(device)
    model.reset_parameters()
    return device, model


def main_func(save_path: str, save_name: str, step: int, under_sampling_threshold: float, model_name: str,
              code_embedding=200, epochs=50, lr=0.001, batch_size=16, hidden_size=64, hidden_size_2=64, out_feats=16, dropout=0.2,
              threshold=0.5, use_gpu=True, load_lazy=True):
    """
    node classification

    :param save_path: 保存模型和结果的路径
    :param save_name: 保存的模型的名字
    :param step: 步长
    :param under_sampling_threshold: 欠采样比例阈值 default: 5.0
    :param model_name: 需要选择的模型： GCN, GAT, GraphSAGE, GGNN default:GCN
    :param code_embedding: 节点嵌入维度 default: 200
    :param epochs: 训练次数 default: 50
    :param lr: 学习率 default: 0.001
    :param batch_size: 批次大小 default: 16
    :param hidden_size: 隐藏层大小 default: 64
    :param out_feats: 输出层大小 default:16,（全连接层的输入），全连接层最终输出维度为 1
    :param threshold: classification threshold
    :param dropout: dropout rate
    :param use_gpu: default True
    :param load_lazy: 是否懒加载图数据，default True
    :return: None
    """
    print(
        f'model: {model_name}, step: {step}, dropout: {dropout}, undersampling: {under_sampling_threshold}, epoch: {epochs}')
    if model_name.startswith('RGCN'):
        self_loop = False
    else:
        self_loop = True
    device, model = init(model_name, code_embedding, hidden_size, hidden_size_2, out_feats, dropout, use_gpu)
    start_train(save_path, save_name, step, under_sampling_threshold, model, device, epochs, lr, batch_size, threshold,
                self_loop, load_lazy)
