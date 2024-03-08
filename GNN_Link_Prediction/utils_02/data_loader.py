import ast
import math
import os
from os.path import join
from typing import Literal

import dgl
import pandas as pd
import torch
from dgl import DGLGraph
from pandas import DataFrame
from torch.utils.data import DataLoader

gnn_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
LOAD_MODE = Literal['train', 'valid', 'test']


def get_graph_files(mode: LOAD_MODE):
    """
    获取图数据保存文件

    :return: [(nodes.tsv, edges.tsv), ...]
    """
    graph_path = join(gnn_path, 'model_dataset_1')
    model_list = os.listdir(graph_path)
    graph_files = []
    model_list = sorted(model_list, key=lambda x: int(x))
    total = len(model_list)
    if mode == 'train':
        start_index = 0
        end_index = int(total * 0.8)
    elif mode == 'valid':
        start_index = int(total * 0.8)
        end_index = int(total * 0.9)
    else:
        start_index = int(total * 0.9)
        end_index = total
    for model_dir in model_list[start_index: end_index]:
        # print('---------------', model_dir)
        if not os.path.exists(join(graph_path, model_dir, 'nodes.tsv')):
            continue
        graph_files.append((join(graph_path, model_dir, 'nodes.tsv'), join(graph_path, model_dir, 'edges.tsv')))
    return graph_files


def random_under_sampling(nodes: DataFrame, edges: DataFrame, mode='train', threshold=4.0):
    """
    随机欠采样，根据阈值随机欠采样，如果超过阈值，则随机采样生成多对结果，count = neg / pos / threshold

    :param nodes: 节点数据 columns=['node_id', 'code_embedding']
    :param edges: 边数据 columns=['start_node_id', 'end_node_id', 'relation', 'label']
    :param threshold: 欠采样阈值 负样本/正样本
    :param mode: 数据集选择 train, valid, test
    :return: 随机欠采样后的节点数据和边数据,可能有多对
    """
    # 验证集和测试集不需要采样
    if mode == 'valid' or mode == 'test':
        return [(nodes, edges)]
    neg_count = (edges['label'] == 0).sum()
    pos_count = (edges['label'] == 1).sum()
    if pos_count == 0:
        pos_count = 1
    # 如果没有达到阈值，不需要采样
    if neg_count / pos_count <= threshold:
        return [(nodes, edges)]
    count = math.floor(neg_count / pos_count / threshold)
    final_graphs = []
    # print(neg_count, pos_count)
    # 循环采样，生成多个图，防止信息过分丢失
    for _ in range(count):
        all_neg_edges = edges[edges['label'] == 0]
        need_sample_num = pos_count * threshold
        sample_neg_edges = all_neg_edges.sample(int(need_sample_num))
        final_edges = pd.concat([edges[edges['label'] == 1], sample_neg_edges])
        merged_ids = pd.concat([final_edges['start_node_id'], final_edges['end_node_id']]).drop_duplicates().tolist()
        final_nodes = nodes[nodes['node_id'].isin(merged_ids)]
        final_nodes.reset_index(drop=True, inplace=True)  # 下面需要根据索引修改，需要重置
        final_edges.reset_index(drop=True, inplace=True)
        # 构建新的节点id映射关系
        map_ids = dict()
        index = 0
        id_list = final_nodes['node_id'].tolist()
        # print('id_list', id_list)
        # 更新节点
        for _id in id_list:
            map_ids[_id] = index
            final_nodes.at[index, 'node_id'] = index
            index += 1
        # 更新边
        for i, row in final_edges.iterrows():
            # 在这里可以添加修改 'column1' 的逻辑
            final_edges.at[i, 'start_node_id'] = map_ids.get(row['start_node_id'])
            final_edges.at[i, 'end_node_id'] = map_ids.get(row['end_node_id'])
        final_graphs.append((final_nodes.copy(), final_edges.copy()))
    return final_graphs


def load_graph_data(node_file, edge_file, mode: LOAD_MODE):
    """
    加载图，从nodes.tsv,edges.tsv文件加载节点特征和边信息

    :param node_file: 节点文件
    :param edge_file: 边文件
    :param mode: 数据集选择
    :return: 图
    """
    # print('handling {0}'.format(node_file))
    nodes = pd.read_csv(node_file)  # columns=['node_id', 'code_embedding']
    edges = pd.read_csv(edge_file)  # columns=['start_node_id', 'end_node_id', 'relation', 'label']
    sample_result = random_under_sampling(nodes, edges, mode)
    # print(f'sample count: {len(sample_result)}')
    graphs = []
    for nodes, edges in sample_result:
        # print('nodes:\n{0} \n edges:\n{1}'.format(nodes, edges))
        src, dst = edges['start_node_id'].tolist(), edges['end_node_id'].tolist()
        relations, labels = torch.tensor(edges['relation'].tolist()), torch.tensor(
            edges['label'].tolist(), dtype=torch.float32)
        g = dgl.graph(data=(src, dst))
        g.ndata['embedding'] = torch.Tensor(nodes['code_embedding'].apply(lambda x: ast.literal_eval(x)).tolist())
        g.edata['label'] = labels
        g.edata['relation'] = relations
        # 添加自环边
        if mode == 'train':
            g = dgl.add_self_loop(g, edge_feat_names=['label'], fill_data=1)
        else:
            g = dgl.add_self_loop(g, edge_feat_names=['label'], fill_data=1)
        # 将所有的自环边的relation设置为 4
        r = g.edata['relation']
        e = g.all_edges()
        for i in range(len(e[0])):
            if e[0][i] == e[1][i]:
                r[i] = 4
        g.edata['relation'] = r
        graphs.append(g)
    # print(g, g.nodes(), g.edges())
    # print(g)
    return graphs


# 构建数据集和 DataLoader
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, gs: list[DGLGraph]):
        self.graphs = gs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        gra = self.graphs[idx]
        features = gra.ndata['embedding']
        labels = gra.edata['label']
        edge_types = gra.edata['relation']
        return gra, features, labels, edge_types


def collate(samples):
    graphs, features, labels, edge_types = map(list, zip(*samples))
    # 使用 dgl.batch 进行批处理，确保正确处理 DGLGraph 对象
    batched_graph = dgl.batch(graphs)
    # 将 features 和 edge_labels 转换为张量
    features = torch.cat(features, dim=0)
    label = torch.cat(labels, dim=0)
    edge_types = torch.cat(edge_types, dim=0)
    return batched_graph, features, label, edge_types


def load_prediction_data(mode: LOAD_MODE, batch_size: int) -> torch.utils.data.dataloader.DataLoader:
    """
    根据模式加载数据集

    :param mode: Literal['train', 'valid', 'test']
    :param batch_size: batch大小
    :return: 相应数据集的DataLoader
    """
    # 从文件加载多个图数据
    graphs = []
    for node_file, edge_file in get_graph_files(mode):
        graphs = graphs + load_graph_data(node_file, edge_file, mode)
    # 创建数据集和 DataLoader
    dataset = GraphDataset(graphs)
    # 这里的batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    return data_loader
