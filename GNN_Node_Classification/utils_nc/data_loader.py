import ast
import math
import os
from os.path import join
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

import dgl
import pandas as pd
import torch
from dgl import DGLGraph
from pandas import DataFrame
from torch.utils.data import DataLoader
import torch.nn.functional as F

LOAD_MODE = Literal['train', 'valid', 'test']


def get_graph_files(dataset_path, mode: LOAD_MODE, embedding_type: str, step: int):
    """
    获取图数据保存文件

    :return: [(nodes.tsv, edges.tsv), ...]
    """
    graph_path = join(dataset_path, f'{embedding_type}_model_dataset_{str(step)}')
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


def compute_similarity(embeddings):
    num_nodes = embeddings.shape[0]
    similarity_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


def similarity_sample(graph: DGLGraph):
    kinds = torch.unique(graph.ndata['kind'])
    # 去除负样本那种和正样本相似度超过平均值的 elements
    for kind in kinds:
        if kind != 0 and kind != 1:
            continue
        kind_nodes = (graph.ndata['kind'] == kind).nonzero(as_tuple=True)[0]
        label_1_nodes = kind_nodes[(graph.ndata['label'][kind_nodes] == 1).nonzero(as_tuple=True)[0]]
        label_0_nodes = kind_nodes[(graph.ndata['label'][kind_nodes] == 0).nonzero(as_tuple=True)[0]]

        avg_label_1_similarity = 0

        if label_1_nodes.numel() > 1:
            label_1_embeddings = graph.ndata['embedding'][label_1_nodes]
            label_1_similarities = compute_similarity(label_1_embeddings)
            avg_label_1_similarity = label_1_similarities[label_1_similarities != 0].mean().item()

        label_0_to_label_1_similarities = []
        for node_0 in label_0_nodes:
            curr_node = []
            for node_1 in label_1_nodes:
                similarity = F.cosine_similarity(graph.ndata['embedding'][node_0].unsqueeze(0),
                                                 graph.ndata['embedding'][node_1].unsqueeze(0), dim=1)
                curr_node.append(similarity.item())
            # (curr_node > avg_label_1_similarity).count()
            if len(curr_node) > 0:
                label_0_to_label_1_similarities.append(sum(curr_node) / len(curr_node))

        if label_0_to_label_1_similarities and label_1_nodes.numel() > 1:
            new_s = label_0_to_label_1_similarities.copy()
            new_s.sort()
            # mid_similarity = new_s[int((len(new_s) / 4) * 3)]
            # average_similarity = sum(label_0_to_label_1_similarities) / len(label_0_to_label_1_similarities)
            # 因为正样本之间的相似度比较高，所以去除那些负样本与正样本相似度比较低的节点，小于正样本平均值的就去掉
            t = torch.tensor(label_0_to_label_1_similarities)
            need_to_remove = label_0_nodes[t < avg_label_1_similarity]
            graph.remove_nodes(need_to_remove.tolist())
    true_nodes = (graph.ndata['label'] == 1).nonzero(as_tuple=True)[0].tolist()
    while True:
        curr_len = len(true_nodes)
        src, dst = graph.edges()
        for i in range(len(src)):
            if src[i].item() in true_nodes and dst[i].item() not in true_nodes:
                true_nodes.append(dst[i].item())
            if dst[i].item() in true_nodes and src[i].item() not in true_nodes:
                true_nodes.append(src[i].item())
        if len(true_nodes) == curr_len:
            break
    all_nodes = graph.nodes()
    need_to_remove = []
    for node in all_nodes:
        if node not in true_nodes:
            need_to_remove.append(node)
    graph.remove_nodes(need_to_remove)
    return graph


def cluster_sample(graph: DGLGraph):
    # 获取所有不同的 kind 值
    kinds = torch.unique(graph.ndata['kind'])
    # 遍历每个 kind 并保留正样本聚类最多的类别的聚类节点
    for kind in kinds:
        if kind != 0 and kind != 1:
            continue
        kind_nodes = (graph.ndata['kind'] == kind).nonzero(as_tuple=True)[0]
        label_1_nodes = kind_nodes[(graph.ndata['label'][kind_nodes] == 1).nonzero(as_tuple=True)[0]]
        label_0_nodes = kind_nodes[(graph.ndata['label'][kind_nodes] == 0).nonzero(as_tuple=True)[0]]

        # 蒋所有节点合并后，进行聚类
        if label_1_nodes.numel() > 0 and label_0_nodes.numel() > 0:
            label_1_embeddings = graph.ndata['embedding'][label_1_nodes]
            label_0_embeddings = graph.ndata['embedding'][label_0_nodes]
            all_embeddings = torch.cat((label_1_embeddings, label_0_embeddings), dim=0).tolist()
            num_clusters = 2
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(all_embeddings)
            labels = kmeans.labels_
            size_1 = label_1_embeddings.shape[0]
            size_0 = label_0_embeddings.shape[0]
            # print(labels)
            # 计算正样本的最多聚类标签
            cluster_labels = np.array(labels[0: size_1])
            # 计算众数
            counter = Counter(cluster_labels)
            mode_data = counter.most_common(1)[0]
            # 获取众数和出现频率
            mode = mode_data[0]
            count = mode_data[1]
            label_1 = labels[0: size_1]
            label_0 = labels[size_1:]
            need_to_remove_0 = label_0_nodes[label_0 != mode]
            if need_to_remove_0.shape[0] != 0:
                graph.remove_nodes(need_to_remove_0.tolist())
            need_to_remove_1 = label_1_nodes[label_1 != mode]
            if need_to_remove_1.shape[0] != 0:
                graph.remove_nodes(need_to_remove_1.tolist())
    true_nodes = (graph.ndata['label'] == 1).nonzero(as_tuple=True)[0].tolist()
    while True:
        curr_len = len(true_nodes)
        src, dst = graph.edges()
        for i in range(len(src)):
            if src[i].item() in true_nodes and dst[i].item() not in true_nodes:
                true_nodes.append(dst[i].item())
            if dst[i].item() in true_nodes and src[i].item() not in true_nodes:
                true_nodes.append(src[i].item())
        if len(true_nodes) == curr_len:
            break
    all_nodes = graph.nodes()
    need_to_remove = []
    for node in all_nodes:
        if node not in true_nodes:
            need_to_remove.append(node)
    graph.remove_nodes(need_to_remove)
    return graph



def my_under_sampling(nodes: DataFrame, edges: DataFrame, mode: str, threshold: float):
    """
    分层随机欠采样（保证图的连通性），根据阈值随机欠采样，如果超过阈值，则随机采样生成多对结果，count = neg / pos / threshold

    :param nodes: 节点数据 columns=['node_id', 'code_embedding', 'label', 'kind']
    :param edges: 边数据 columns=['start_node_id', 'end_node_id', 'relation']
    :param threshold: 欠采样阈值 负样本/正样本
    :param mode: 样本选择，train, valid, test
    :return: 随机欠采样后的节点数据和边数据,可能有多对
    """
    # print('threshold = ', threshold)
    # 验证集和测试集固定采样比,也就是峰值，训练集之后在此基础上进行网格搜索
    if mode == 'valid' or mode == 'test':
        threshold = 30.0
    neg_count = (nodes['label'] == 0).sum()
    pos_count = (nodes['label'] == 1).sum()
    if pos_count == 0:
        pos_count = 1
    # 如果没有达到阈值，不需要采样
    if neg_count / pos_count <= threshold:
        return [(nodes, edges)]
    # 最多采样 1 次
    count = min(math.floor(neg_count / pos_count / threshold), 1)
    final_graphs = []
    # print(neg_count, pos_count)
    all_neg_nodes = nodes[nodes['label'] == 0]
    need_sample_num = pos_count * threshold
    # 循环不重复采样，生成多个图，防止信息过分丢失
    for _ in range(count):
        final_nodes = nodes[(nodes['label'] == 1)]
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


def random_under_sampling(graph, mode, under_sampling_threshold):
    # 提取第一个维度的数据并转成 DataFrame 的列
    data = {
        'node_id': [graph.nodes()[i].item() for i in range(graph.nodes().shape[0])],
        'embedding': [graph.ndata['embedding'][i].tolist() for i in range(graph.ndata['embedding'].shape[0])],
        'label': [graph.ndata['label'][i].item() for i in range(graph.ndata['label'].shape[0])],
        'kind': [graph.ndata['kind'][i].item() for i in range(graph.ndata['kind'].shape[0])]
    }
    # 创建 DataFrame
    nodes = pd.DataFrame(data)
    (src, dst) = graph.edges()
    data = {
        'start_node_id': [src[i].item() for i in range(src.shape[0])],
        'end_node_id': [dst[i].item() for i in range(dst.shape[0])],
        'relation': [graph.edata['relation'][i].item() for i in range(graph.edata['relation'].shape[0])]
    }
    # 创建 DataFrame
    edges = pd.DataFrame(data)
    sample_nodes, sample_edges = my_under_sampling(nodes, edges, mode, under_sampling_threshold)[0]
    src, dst = sample_edges['start_node_id'].tolist(), sample_edges['end_node_id'].tolist()
    g = dgl.graph(data=(src, dst), num_nodes=len(sample_nodes))
    g.ndata['embedding'] = torch.Tensor(sample_nodes['embedding'].tolist())
    g.ndata['label'] = torch.tensor(sample_nodes['label'].tolist(), dtype=torch.float32)
    g.ndata['kind'] = torch.tensor(sample_nodes['kind'].tolist(), dtype=torch.int64)
    g.edata['relation'] = torch.tensor(sample_edges['relation'].tolist(), dtype=torch.int64)
    return g


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
    nodes = pd.read_csv(node_file)  # columns=['node_id', 'code_embedding', 'label', 'kind']
    edges = pd.read_csv(edge_file)  # columns=['start_node_id', 'end_node_id', 'relation']
    graphs = []
    src, dst = edges['start_node_id'].tolist(), edges['end_node_id'].tolist()
    g = dgl.graph(data=(src, dst), num_nodes=len(nodes))
    g.ndata['embedding'] = torch.Tensor(nodes['code_embedding'].apply(lambda x: ast.literal_eval(x)).tolist())
    g.ndata['label'] = torch.tensor(nodes['label'].tolist(), dtype=torch.float32)
    kind_mapping = {'variable': 0, 'function': 1, 'class': 2, 'interface': 3}
    nodes['kind_encoded'] = nodes['kind'].map(kind_mapping)
    g.ndata['kind'] = torch.tensor(nodes['kind_encoded'].tolist(), dtype=torch.int64)
    g.edata['relation'] = torch.tensor(edges['relation'].tolist(), dtype=torch.int64)
    # 是否 similarity 过滤
    # if under_sampling_threshold == 0:
    # g = similarity_sample(g)
    g = cluster_sample(g)
    # 是否 undersample
    if under_sampling_threshold > 0:
        g = random_under_sampling(g, mode, under_sampling_threshold)
    # 添加自环边
    if self_loop:
        g = dgl.add_self_loop(g, edge_feat_names=['relation'], fill_data=4)
    # 是否使用 similarity 过滤
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
        # seeds = gra.ndata['seed']
        kinds = gra.ndata['kind'] if 'kind' in gra.ndata.keys() else gra.ndata['label']
        return gra, features, labels, edge_types, kinds


def collate(batch):
    """
    合并一个样本列表，形成一个小批量的张量。当从map样式数据集批量加载时使用。

    :param batch: batch图集合
    :return: 合并后的大图，以及特征集合
    """
    graphs, features, labels, edge_types, kinds = map(list, zip(*batch))
    # 使用 dgl.batch 进行批处理，确保正确处理 DGLGraph 对象,实际就是将多个图合并为一个大图
    batched_graph = dgl.batch(graphs)
    # 将 features 和 edge_labels 转换为张量
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    # seeds = torch.cat(seeds, dim=0)
    kinds = torch.cat(kinds, dim=0)
    edge_types = torch.cat(edge_types, dim=0)
    return batched_graph, features, labels, edge_types, kinds


def load_prediction_data(dataset_path, mode: LOAD_MODE, embedding_type: str, batch_size: int, step: int, under_sampling_threshold=15,
                         self_loop=True, load_lazy=True) -> torch.utils.data.dataloader.DataLoader:
    """
    根据模式加载数据集,可以选择懒加载

    :param dataset_path: 数据集保存路径
    :param step: 步长
    :param embedding_type: type of embedding
    :param mode: Literal['train', 'valid', 'test']
    :param batch_size: batch大小
    :param under_sampling_threshold: 欠采样阈值，最终负/正样本比例
    :param self_loop: 是否需要添加自环
    :param load_lazy: 是否加载之前的数据
    :return: 相应数据集的DataLoader
    """
    old_data_path = join(dataset_path, f'{embedding_type}_model_dataset_{str(step)}', f'{mode}',
                         f'old_data_{mode}_{under_sampling_threshold}.pkl')
    if load_lazy and os.path.exists(old_data_path):
        print('lazyload...')
        preloaded_dataset = pd.read_pickle(old_data_path)
    else:
        # 从文件加载多个图数据
        graphs = []
        for node_file, edge_file in get_graph_files(dataset_path, mode, embedding_type, step):
            graphs = graphs + load_graph_data(node_file, edge_file, mode, under_sampling_threshold, self_loop)
        # 创建数据集和 DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preloaded_dataset = PreloadedGraphDataset(graphs, device)
        pd.to_pickle(preloaded_dataset, old_data_path)
    print(f'total graph: {len(preloaded_dataset)}')
    shuffle = True if mode == 'train' else False
    data_loader = DataLoader(preloaded_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return data_loader
