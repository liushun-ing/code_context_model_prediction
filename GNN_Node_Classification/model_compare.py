"""
比对 model 中的正样本之间是否特征相似等，发现规律
"""
import ast
import os
from os.path import join

import dgl
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torch.nn.functional as F

import torch

from dataset_split_util import get_models_by_ratio

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')

edge_vector = {
    "declares": 0,
    "calls": 1,
    "inherits": 2,
    "implements": 3,
}

node_kind_list = ['variable', 'function', 'class', 'interface']


def get_graph_list(ratio, step=1, project_model_name='my_mylyn'):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = get_models_by_ratio(project_model_name, ratio[0], ratio[1])
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    embedding_file_1 = f'mylyn_1_astnn_embedding.pkl'
    embedding_file_2 = f'mylyn_1_codebert_embedding.pkl'
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    graph_list = []
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'new_1_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读 embedding
        embedding_list_1 = pd.read_pickle(join(model_path, embedding_file_1))
        embedding_list_2 = pd.read_pickle(join(model_path, embedding_file_2))
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        if len(graphs) == 0:
            continue
        for graph in graphs:
            # 创建保存nodes和edges的数据结构
            nodes_tsv = list()
            edges_tsv = list()
            true_node = 0
            true_edge = 0
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')])
                _id = int(vertex.get('id'))
                # 去找 embedding 并添加  这儿不知道为什么会多一层列表
                # print(node_id)
                embedding_1 = embedding_list_1[embedding_list_1['id'] == node_id]['embedding'].iloc[0]
                embedding_2 = embedding_list_2[embedding_list_2['id'] == node_id]['embedding'].iloc[0]
                # 进行组合
                embedding = embedding_1.tolist() + embedding_2.tolist()
                # embedding = embedding_1.tolist()
                nodes_tsv.append([_id, embedding, int(vertex.get('origin')), vertex.get('kind')])
                if int(vertex.get('origin')) == 1:
                    true_node += 1
            edges = graph.find('edges')
            edge_list = edges.findall('edge')
            for edge in edge_list:
                start = int(edge.get('start'))
                end = int(edge.get('end'))
                label = edge.get('label')
                edges_tsv.append([start, end, edge_vector[label]])
                if int(edge.get('origin')) == 1:
                    true_edge += 1
            if len(nodes_tsv) > 0 and len(edges_tsv) > 0:
                if true_node > step:
                    node_df = pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding', 'label', 'kind'])
                    edge_df = pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation'])
                    src, dst = edge_df['start_node_id'].tolist(), edge_df['end_node_id'].tolist()
                    g = dgl.graph(data=(src, dst), num_nodes=len(node_df))
                    g.ndata['embedding'] = torch.Tensor(node_df['code_embedding'].tolist())
                    g.ndata['label'] = torch.tensor(node_df['label'].tolist(), dtype=torch.float32)
                    kind_mapping = {'variable': 0, 'function': 1, 'class': 2, 'interface': 3}
                    node_df['kind_encoded'] = node_df['kind'].map(kind_mapping)
                    g.ndata['kind'] = torch.tensor(node_df['kind_encoded'].tolist(), dtype=torch.int64)
                    g.edata['relation'] = torch.tensor(edge_df['relation'].tolist(), dtype=torch.int64)
                    graph_list.append(g)
    return graph_list


def compute_similarity(embeddings):
    num_nodes = embeddings.shape[0]
    similarity_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


# 统计每行的平均值，也就是每个节点与其他节点的相似度平均值
def extract_upper_triangle(matrix):
    rows, cols = matrix.shape
    upper_triangle_elements = []

    for i in range(rows):
        curr_node = []
        for j in range(cols):
            if i != j:
                curr_node.append(matrix[i, j].item())
        upper_triangle_elements.append(sum(curr_node) / len(curr_node))

    return upper_triangle_elements


if os.path.exists('graph_list_train.pkl'):
    graph_list = pd.read_pickle('graph_list_train.pkl')
else:
    graph_list = get_graph_list(ratio=[0, 0.8])
    pd.to_pickle(graph_list, 'graph_list_train.pkl')

print(len(graph_list))
# similarity_list = [[], [], [], []]
#
# for g in graph_list:
#     print(f"{g.num_nodes()} nodes")
#     # 获取所有不同的 kind 值
#     kinds = torch.unique(g.ndata['kind'])
#     # 遍历每个 kind 并计算相似性矩阵
#     for kind in kinds:
#         same_kind_nodes = (g.ndata['kind'] == kind).nonzero(as_tuple=True)[0]
#         if same_kind_nodes.numel() > 1:  # 至少需要两个节点计算相似性
#             same_kind_embeddings = g.ndata['embedding'][same_kind_nodes]
#             similarities = compute_similarity(same_kind_embeddings)
#             average_similarity = similarities[similarities != 0].mean().item()
#             similarity_list[kind.item()].append(average_similarity)
#             print(f"Average similarity for nodes with kind {node_kind_list[kind.item()]}: {average_similarity:.4f}")
#             print(torch.mean(similarities, dim=1))
#
# print('----------------------------------')
# for i, similarity in enumerate(similarity_list):
#     if len(similarity) > 0:
#         print(f"Total Average similarity for nodes with kind {node_kind_list[i]}: {np.average(similarity):.4f}")

label_1_sum = []
label_10_sum = []
all_1_field = []
all_1_method = []
all_field = []
all_method = []
for g in graph_list:
    print(f"{g.num_nodes()} nodes")
    # 获取所有不同的 kind 值
    kinds = torch.unique(g.ndata['kind'])

    # 遍历每个 kind 并计算相似性
    for kind in kinds:
        kind_nodes = (g.ndata['kind'] == kind).nonzero(as_tuple=True)[0]
        label_1_nodes = kind_nodes[(g.ndata['label'][kind_nodes] == 1).nonzero(as_tuple=True)[0]]
        label_0_nodes = kind_nodes[(g.ndata['label'][kind_nodes] == 0).nonzero(as_tuple=True)[0]]

        # 计算 label 为 1 的节点之间的相似性
        if label_1_nodes.numel() > 1:
            label_1_embeddings = g.ndata['embedding'][label_1_nodes]
            label_1_similarities = compute_similarity(label_1_embeddings)
            avg_label_1_similarity = label_1_similarities[label_1_similarities != 0].mean().item()
            if kind == 1:
                label_1_sum.append(avg_label_1_similarity)
            if kind == 0:
                all_1_field += extract_upper_triangle(label_1_similarities)
            elif kind == 1:
                all_1_method += extract_upper_triangle(label_1_similarities)
            print(f"Kind {node_kind_list[kind.item()]}: Average similarity for nodes with label 1: {avg_label_1_similarity:.4f}")

        # 计算 label 为 1 的节点与 label 为 0 的节点之间的相似性
        label_1_to_label_0_similarities = []
        for node_0 in label_0_nodes:
            curr_node = []
            for node_1 in label_1_nodes:
                similarity = F.cosine_similarity(g.ndata['embedding'][node_0].unsqueeze(0),
                                                 g.ndata['embedding'][node_1].unsqueeze(0), dim=1)
                curr_node.append(similarity.item())
            if len(curr_node) > 0:
                label_1_to_label_0_similarities.append(sum(curr_node) / len(curr_node))

        if label_1_to_label_0_similarities:
            avg_label_1_to_label_0_similarity = sum(label_1_to_label_0_similarities) / len(label_1_to_label_0_similarities)
            if kind == 1:
                label_10_sum.append(avg_label_1_to_label_0_similarity)
            if kind == 0:
                all_field += label_1_to_label_0_similarities
            elif kind == 1:
                all_method += label_1_to_label_0_similarities
            print(label_1_to_label_0_similarities)
            print(
                f"Kind {node_kind_list[kind.item()]}: Average similarity between nodes with label 1 and nodes with label 0: {avg_label_1_to_label_0_similarity:.4f}")

print(f"\n所有 label1 的平均值{np.mean(label_1_sum)}")
print(f"所有 label10 的平均值{np.mean(label_10_sum)}\n")

# 定义起始区间、结束区间和间隔
start = 0.90
end = 1
interval = 0.01

# 计算区间边界
bins = np.arange(start, end + interval, interval)

# 计算每个区间的统计量
hist, bin_edges = np.histogram(all_1_field, bins=bins)
print('label1 variable')
print(len(all_1_field))
# 打印每个区间的统计量
for i in range(len(hist)):
    print(f"Bin {i + 1}: {hist[i]} values in range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})")
print(f"below 0.9: {len(all_1_field) - sum(hist)}")

# 计算每个区间的统计量
hist, bin_edges = np.histogram(all_1_method, bins=bins)
print('label1 method')
print(len(all_1_method))
# 打印每个区间的统计量
for i in range(len(hist)):
    print(f"Bin {i + 1}: {hist[i]} values in range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})")
print(f"below 0.9: {len(all_1_method) - sum(hist)}")

print('label10 variable')
# 计算每个区间的统计量
hist, bin_edges = np.histogram(all_field, bins=bins)
print(len(all_field))
# 打印每个区间的统计量
for i in range(len(hist)):
    print(f"Bin {i + 1}: {hist[i]} values in range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})")
print(f"below 0.9: {len(all_field) - sum(hist)}")

print('label10 function')
# 计算每个区间的统计量
hist, bin_edges = np.histogram(all_method, bins=bins)
print(len(all_method))
# 打印每个区间的统计量
for i in range(len(hist)):
    print(f"Bin {i + 1}: {hist[i]} values in range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})")
print(f"below 0.9: {len(all_method) - sum(hist)}")



