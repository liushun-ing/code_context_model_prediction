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

LOAD_MODE = Literal['train', 'valid', 'test']


def get_graph_files(dataset_path, mode: LOAD_MODE, step: int):
    """
    获取图数据保存文件

    :return: [(nodes.tsv, edges.tsv), ...]
    """
    graph_path = join(dataset_path, f'model_dataset_{str(step)}')
    graph_files = []
    if mode == 'train':
        if os.path.exists(join(graph_path, 'train')):
            model_list = os.listdir(join(graph_path, 'train'))
        else:
            model_list = []
    elif mode == 'valid':
        if os.path.exists(join(graph_path, 'valid')):
            model_list = os.listdir(join(graph_path, 'valid'))
        else:
            model_list = []
    else:
        if os.path.exists(join(graph_path, 'test')):
            model_list = os.listdir(join(graph_path, 'test'))
        else:
            model_list = []
    for model_dir in model_list:
        # if not model_dir.startswith('my_mylyn_5834_0'):
        #     continue
        # print(model_dir)
        if not os.path.exists(join(graph_path, mode, model_dir, 'nodes.tsv')):
            continue
        graph_files.append(
            (join(graph_path, mode, model_dir, 'nodes.tsv'), join(graph_path, mode, model_dir, 'edges.tsv')))
    return graph_files


def my_under_sampling(nodes: DataFrame, edges: DataFrame, mode: str, threshold: float):
    """
    分层随机欠采样（保证图的连通性），根据阈值随机欠采样，如果超过阈值，则随机采样生成多对结果，count = neg / pos / threshold

    :param nodes: 节点数据 columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']
    :param edges: 边数据 columns=['start_node_id', 'end_node_id', 'relation']
    :param threshold: 欠采样阈值 负样本/正样本
    :param mode: 样本选择，train, valid, test
    :return: 随机欠采样后的节点数据和边数据,可能有多对
    """
    # 验证集和测试集固定采样比，训练集之后在此基础上进行网格搜索
    if mode == 'valid' or mode == 'test':
        threshold = 30.0
    if threshold == 0:  # 阈值为0不用采样
        return [(nodes, edges)]
    neg_count = (nodes['label'] == 0).sum()
    pos_count = (nodes['label'] == 1).sum()
    if pos_count == 0:
        pos_count = 1
    # 如果没有达到阈值，不需要采样
    if neg_count / pos_count <= threshold:
        return [(nodes, edges)]
    # 最多采样 5 次
    count = min(math.floor(neg_count / pos_count / threshold), 1)
    final_graphs = []
    # print(neg_count, pos_count)
    all_neg_nodes = nodes[nodes['label'] == 0]
    need_sample_num = pos_count * threshold
    # 循环不重复采样，生成多个图，防止信息过分丢失
    for _ in range(count):
        final_nodes = nodes[(nodes['label'] == 1) | (nodes['seed'] == 1)]
        sample_count = need_sample_num
        while len(final_nodes) < (pos_count + need_sample_num):
            # print(len(final_nodes), ' ', pos_count + need_sample_num, ' ', sample_count)
            sample_neg_nodes = all_neg_nodes.sample(int(sample_count))
            # 只留下与final_nodes中的节点有关联的节点
            middle_ids = final_nodes['node_id'].tolist()
            middle_edges = edges[edges['start_node_id'].isin(middle_ids) | edges['end_node_id'].isin(middle_ids)]
            t_nodes = list(set(middle_edges['start_node_id'].tolist() + middle_edges['end_node_id'].tolist()))
            sample_neg_nodes = sample_neg_nodes[sample_neg_nodes['node_id'].isin(t_nodes)]
            sample_count -= len(sample_neg_nodes)
            final_nodes = pd.concat([final_nodes, sample_neg_nodes])
        merged_ids = final_nodes['node_id'].tolist()
        final_edges = edges[edges['start_node_id'].isin(merged_ids) & edges['end_node_id'].isin(merged_ids)]
        final_nodes = final_nodes.copy(deep=True)
        final_edges = final_edges.copy(deep=True)
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
            final_edges.at[i, 'start_node_id'] = map_ids.get(row['start_node_id'])
            final_edges.at[i, 'end_node_id'] = map_ids.get(row['end_node_id'])
        final_graphs.append((final_nodes, final_edges))
    return final_graphs


def load_graph_data(node_file, edge_file, mode: LOAD_MODE, under_sampling_threshold: float, self_loop=True):
    """
    加载图，从nodes.tsv,edges.tsv文件加载节点特征和边信息

    :param node_file: 节点文件
    :param edge_file: 边文件
    :param mode: 数据集模式 Literal['train', 'valid', 'test']
    :param under_sampling_threshold: 欠采样阈值
    :param self_loop: self loop
    :return: 图
    """
    # print('handling {0}'.format(node_file))
    _nodes = pd.read_csv(node_file)  # columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']
    _edges = pd.read_csv(edge_file)  # columns=['start_node_id', 'end_node_id', 'relation', 'label']
    sample_result = my_under_sampling(_nodes, _edges, mode, threshold=under_sampling_threshold)
    # print(f'sample count: {len(sample_result)}')
    graphs = []
    for nodes, edges in sample_result:
        # print('nodes:\n{0} \n edges:\n{1}'.format(nodes, edges))
        src, dst = edges['start_node_id'].tolist(), edges['end_node_id'].tolist()
        g = dgl.graph(data=(src, dst), num_nodes=len(nodes))
        # g = dgl.graph(data=(src + dst, dst + src), num_nodes=len(nodes))  # double edge
        g.ndata['embedding'] = torch.Tensor(nodes['code_embedding'].apply(lambda x: ast.literal_eval(x)).tolist())
        g.ndata['label'] = torch.tensor(nodes['label'].tolist(), dtype=torch.float32)
        kind_mapping = {'variable': 0, 'function': 1, 'class': 2, 'interface': 3}
        nodes['kind_encoded'] = nodes['kind'].map(kind_mapping)
        g.ndata['kind'] = torch.tensor(nodes['kind_encoded'].tolist(), dtype=torch.int64)
        g.ndata['seed'] = torch.tensor(nodes['seed'].tolist(), dtype=torch.float32)
        # g.edata['relation'] = torch.tensor(edges['relation'].tolist() + edges['relation'].tolist(), dtype=torch.int64)
        g.edata['relation'] = torch.tensor(edges['relation'].tolist(), dtype=torch.int64)
        # 添加自环边
        if self_loop:
            g = dgl.add_self_loop(g, edge_feat_names=['relation'], fill_data=5)
        graphs.append(g)
    # print(g, g.nodes(), g.edges())
    return graphs


# 构建数据集和 DataLoader
class PreloadedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, gs: list[DGLGraph], device):
        self.graphs = [graph.to(device) for graph in gs]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        gra = self.graphs[idx]
        features = gra.ndata['embedding']
        labels = gra.ndata['label']
        edge_types = gra.edata['relation']
        seeds = gra.ndata['seed']
        kinds = gra.ndata['kind'] if 'kind' in gra.ndata.keys() else gra.ndata['label']
        return gra, features, labels, edge_types, seeds, kinds


def collate(batch):
    """
    合并一个样本列表，形成一个小批量的张量。当从map样式数据集批量加载时使用。

    :param batch: batch图集合
    :return: 合并后的大图，以及特征集合
    """
    graphs, features, labels, edge_types, seeds, kinds = map(list, zip(*batch))
    # 使用 dgl.batch 进行批处理，确保正确处理 DGLGraph 对象,实际就是将多个图合并为一个大图
    batched_graph = dgl.batch(graphs)
    # 将 features 和 edge_labels 转换为张量
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    seeds = torch.cat(seeds, dim=0)
    kinds = torch.cat(kinds, dim=0)
    edge_types = torch.cat(edge_types, dim=0)
    return batched_graph, features, labels, edge_types, seeds, kinds


def load_prediction_data(dataset_path, mode: LOAD_MODE, batch_size: int, step: int, under_sampling_threshold=5.0,
                         self_loop=True, load_lazy=True) -> torch.utils.data.dataloader.DataLoader:
    """
    根据模式加载数据集,可以选择懒加载

    :param dataset_path: 数据集保存路径
    :param step: 步长
    :param mode: Literal['train', 'valid', 'test']
    :param batch_size: batch大小
    :param under_sampling_threshold: 欠采样阈值，最终负/正样本比例
    :param self_loop: 是否需要添加自环
    :param load_lazy: 是否加载之前的数据
    :return: 相应数据集的DataLoader
    """
    old_data_path = join(dataset_path, f'model_dataset_{str(step)}', f'{mode}',
                         f'old_data_{mode}_{under_sampling_threshold}.pkl')
    if load_lazy and os.path.exists(old_data_path):
        print('lazyload...')
        preloaded_dataset = pd.read_pickle(old_data_path)
    else:
        # 从文件加载多个图数据
        graphs = []
        for node_file, edge_file in get_graph_files(dataset_path, mode, step):
            graphs = graphs + load_graph_data(node_file, edge_file, mode, under_sampling_threshold, self_loop)
        # 创建数据集和 DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preloaded_dataset = PreloadedGraphDataset(graphs, device)
        pd.to_pickle(preloaded_dataset, old_data_path)
    shuffle = True if mode == 'train' else False
    data_loader = DataLoader(preloaded_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return data_loader
