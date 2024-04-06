import os
from os.path import join
import random

import networkx as nx
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
from networkx.algorithms import isomorphism
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

# from dataset_split_util import get_models_by_ratio

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def get_models_by_ratio(project: str, start_ratio: float, end_ratio: float):
    """
    返回比例范围内的model，根据first_time排序

    :param project: 项目
    :param start_ratio: 开始比例
    :param end_ratio: 结束比例
    :return: model_dir数组
    """
    if start_ratio == end_ratio:
        return []
    project_path = join(root_path, project, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    all_models = []
    for model_dir in model_dir_list:
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        first_time = code_context_model.get('first_time')
        all_models.append([model_dir, first_time])
    all_models = sorted(all_models, key=lambda x: x[1])
    m = np.array(all_models)
    return m[int(len(m) * start_ratio):int(len(m) * end_ratio), 0]


def load_patterns(patterns):
    G2s = []
    with open(patterns) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('t #'):
                g = nx.DiGraph()
            if line.startswith('v'):
                v = line.split(' ')
                g.add_node(int(v[1]), label=v[2])
            if line.startswith('e'):
                e = line.split(' ')
                g.add_edge(int(e[1]), int(e[2]), label=e[3])
            if line.startswith('#'):
                G2s.append(g)
        f.close()
    return G2s


def get_graph(graphs: list[ET.Element], step: int):
    gs = []
    for graph in graphs:
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        edges = graph.find('edges')
        edge_list = edges.findall('edge')
        g = nx.DiGraph()
        true_node = 0
        true_edge = 0
        # 转化为图结构
        for node in vertex_list:
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=node.get('origin'))
            if int(node.get('origin')) == 1:
                true_node += 1
        for link in edge_list:
            g.add_edge(int(link.get('start')), int(link.get('end')), label=link.get('label'))
            if int(link.get('origin')) == 1:
                true_edge += 1
        if true_edge > 0 and true_node > step:
            gs.append(g)
    return gs


def load_targets(project_model_name: str, step):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    G1s = []
    # 读取code context model
    model_dir_list = get_models_by_ratio(project_model_name, 0.84, 1)
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        # print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        tree = ET.parse(model_file)  # 拿到xml树
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        G1s = G1s + get_graph(graphs, step)
    return G1s


def count_positive(confidence, thres):
    count = 0
    for con in confidence:
        if con[1] >= thres:
            count += 1
    return count


def node_match(node1, node2):
    return node1['label'] == node2['label']


def edge_match(edge1, edge2):
    return edge1['label'] == edge2['label']


def calculate_result(labels, true_number):
    precision = labels.count(1) / len(labels)
    recall = labels.count(1) / true_number
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # print([precision, recall, f1])
    return [precision, recall, f1]


def calculate_result_full(labels, output, MinConf, true_number):
    result = []
    for MinConf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if len(labels) == 0:
            return [0, 0, 0]
        positive_count = 0
        true_positive_count = 0
        for i in range(len(output)):
            if output[i] >= MinConf:
                positive_count += 1
                if labels[i] == 1:
                    true_positive_count += 1
        precision = true_positive_count / positive_count if positive_count != 0 else 0
        recall = true_positive_count / true_number
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        # print([precision, recall, f1])
        result.append([precision, recall, f1])
    return result


def print_result(result, k):
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for minConf in s:
        print(f'minConf: {minConf}:')
        i = s.index(minConf)
        p, r, f = 0.0, 0.0, 0.0
        for res in result:
            p += res[i][0]
            r += res[i][1]
            f += res[i][2]
        print(f'----------result of top {k}-------\n'
              f'Precision: {p / len(result)}, '
              f'Recall: {r / len(result)}, '
              f'F1: {f / len(result)}')


def main_func(step, MinConf, patterns):
    r = []
    G1s = load_targets('my_mylyn', step)
    G2s = load_patterns(patterns)
    print('G1s', len(G1s), 'G2s', len(G2s))
    result_1, result_3, result_5, result_full = [], [], [], []
    _index = 1200
    for G1 in G1s[900: _index]:
        if G1s.index(G1) in [689,752]:
            continue
        print(f'handling: {G1s.index(G1)}-{G1}')
        total_match = 0
        confidence = dict()
        for G2 in G2s:
            GM = isomorphism.DiGraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
            if GM.subgraph_is_isomorphic():
                for sub_iter in GM.subgraph_isomorphisms_iter():
                    total_match += 1
                    nodes = list(map(int, list(sub_iter.keys())))
                    for node in nodes:
                        if confidence.get(node):
                            confidence[node] = confidence[node] + 1
                        else:
                            confidence[node] = 1
                    # sub_G1: nx.DiGraph = G1.subgraph(nodes)
                    # print(sub_G1.nodes.data(), sub_G1.edges.data())
        for i in confidence:
            confidence[i] = confidence.get(i) / total_match
        confidence = sorted(confidence.items(), key=lambda d: d[1], reverse=True)
        # print(f'{G1} confidence {confidence}')
        r.append(count_positive(confidence, MinConf))
        k = 1
        if k > 0:
            length = len(confidence)
            topk = min(len(G1.nodes), k)
            labels = []
            if (length >= topk):
                top_confidence = confidence[:topk]
                for top_c in top_confidence:
                    labels.append(int(G1.nodes.get(top_c[0])['origin']))
            else:
                top_confidence = confidence[:]
                for top_c in top_confidence:
                    # print(G1.nodes.get(node_id))
                    labels.append(int(G1.nodes.get(top_c[0])['origin']))
                # random select topk-length
                for i in range(topk - length):
                    while True:
                        _id = random.choice(list(G1.nodes))
                        if _id not in top_confidence:
                            labels.append(int(G1.nodes.get(_id)['origin']))
                            break
            true_number = 0
            for n in list(G1.nodes):
                true_number += int(G1.nodes.get(n)['origin'])
            result_1.append(calculate_result(labels, true_number))
        k = 3
        if k > 0:
            length = len(confidence)
            topk = min(len(G1.nodes), k)
            labels = []
            if (length >= topk):
                top_confidence = confidence[:topk]
                for top_c in top_confidence:
                    labels.append(int(G1.nodes.get(top_c[0])['origin']))
            else:
                top_confidence = confidence[:]
                for top_c in top_confidence:
                    # print(G1.nodes.get(node_id))
                    labels.append(int(G1.nodes.get(top_c[0])['origin']))
                # random select topk-length
                for i in range(topk - length):
                    while True:
                        _id = random.choice(list(G1.nodes))
                        if _id not in top_confidence:
                            labels.append(int(G1.nodes.get(_id)['origin']))
                            break
            true_number = 0
            for n in list(G1.nodes):
                true_number += int(G1.nodes.get(n)['origin'])
            result_3.append(calculate_result(labels, true_number))
        k = 5
        if k > 0:
            length = len(confidence)
            topk = min(len(G1.nodes), k)
            labels = []
            if (length >= topk):
                top_confidence = confidence[:topk]
                for top_c in top_confidence:
                    labels.append(int(G1.nodes.get(top_c[0])['origin']))
            else:
                top_confidence = confidence[:]
                for top_c in top_confidence:
                    # print(G1.nodes.get(node_id))
                    labels.append(int(G1.nodes.get(top_c[0])['origin']))
                # random select topk-length
                for i in range(topk - length):
                    while True:
                        _id = random.choice(list(G1.nodes))
                        if _id not in top_confidence:
                            labels.append(int(G1.nodes.get(_id)['origin']))
                            break
            true_number = 0
            for n in list(G1.nodes):
                true_number += int(G1.nodes.get(n)['origin'])
            result_5.append(calculate_result(labels, true_number))
        k = 0
        if k == 0:
            output, labels = [], []
            for top_c in confidence:
                output.append(top_c[1])
                node_id = top_c[0]
                # print(G1.nodes.get(node_id))
                labels.append(int(G1.nodes.get(node_id)['origin']))
            true_number = 0
            for n in list(G1.nodes):
                true_number += int(G1.nodes.get(n)['origin'])
            result_full.append(calculate_result_full(labels, output, MinConf, true_number))
    # print_result(result_1, 1)
    # print_result(result_3, 3)
    # print_result(result_5, 5)
    # print_result(result_full, 0)
    pd.to_pickle(result_full, f'./result-{_index/300}')
    # final_precision = final_precision / len(G1s)
    # final_recall = final_recall / len(G1s)
    # final_f1 = final_f1 / len(G1s)
    # print(f'----------final result-------\n'
    #       f'final_precision: {final_precision}, '
    #       f'final_recall: {final_recall}, '
    #       f'final_f1: {final_f1}')
    # print(r)

# print(111111111)

# G2s = load_patterns()
# print('G2s', len(G2s))
step = 1
patterns = './patterns-0.01'
if step == 1:
    MinConf = 0.3
elif step == 2:
    MinConf = 0.6
else:
    MinConf = 0.1
# for patterns in ['./patterns', './patterns-0.01']:
for patterns in ['./patterns-0.008']:
    print('patterns-----', patterns)
    # for MinConf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print('MinConf------', MinConf)
    main_func(step=step, MinConf=MinConf, patterns=patterns)
# main_func(step=2)
# main_func(step=3)
