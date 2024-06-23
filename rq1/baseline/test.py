import copy
import itertools
import os
import sys
import time
from decimal import Decimal
from os.path import join
import math
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from networkx import DiGraph
from networkx.algorithms import isomorphism
from gspan_mining.config import parser
from gspan_mining.main import main

root_path = join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'params_validation',
                 'git_repo_code')


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
    if len(graphs) == 0:
        return gs
    for graph in graphs:
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        edges = graph.find('edges')
        edge_list = edges.findall('edge')
        g = nx.DiGraph()
        true_node = 0
        # true_edge = 0
        # 转化为图结构
        remove_nodes = []
        for node in vertex_list:
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=node.get('origin'),
                       G1=node.get('G1'))
            if node.get('stereotype') == 'NOTFOUND':
                remove_nodes.append(int(node.get('id')))
            else:
                if int(node.get('origin')) == 1:
                    true_node += 1
        for link in edge_list:
            g.add_edge(int(link.get('start')), int(link.get('end')), label=link.get('label'))
            # if int(link.get('origin')) == 1:
            #     true_edge += 1
        if true_node > step:
            for node_id in remove_nodes:
                g.remove_node(node_id)  # 会自动删除边
            gs.append(g)
    return gs


def load_targets(project_model_name: str, step):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    G1s = []
    # 读取code context model
    model_dir_list = get_models_by_ratio(project_model_name, 0.9, 1)
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        # print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'new_{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        tree = ET.parse(model_file)  # 拿到xml树
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        G1s = G1s + get_graph(graphs, step)
    # for g in G1s:
    #     true_number = 0
    #     for n in list(g.nodes):
    #         true_number += int(g.nodes.get(n)['origin'])
    #     if true_number == len(list(g.nodes)):
    #         print('no change ', G1s.index(g))
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
    if len(labels) == 0:
        return [0, 0, 0]
    precision = labels.count(1) / len(labels)
    recall = labels.count(1) / true_number
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # print([precision, recall, f1])
    return [precision, recall, f1]


def calculate_result_full(labels, output, true_number):
    result = []
    for MinConf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if len(labels) == 0:
            result.append([0, 0, 0])
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
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    for minConf in s:
        i = s.index(minConf)
        p, r, f = 0.0, 0.0, 0.0
        for res in result:
            p += res[i][0]
            r += res[i][1]
            f += res[i][2]
        p = Decimal(p / len(result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        r = Decimal(r / len(result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        # f = Decimal(f / len(result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        if p + r > 0:
            f = Decimal(2 * p * r / (p + r)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        print(f"{minConf:>10.1f} {p:>10.3f} {r:>10.3f} {f:>10.3f}")


def save_result_stereotype(confidence, nodes):
    print("----new match---")
    confidence = dict(confidence)
    for node_id in nodes:
        origin_label = nodes.get(node_id)['origin']
        if confidence.get(node_id):
            conf = confidence.get(node_id)
            # node_id = c[0]
            # conf = Decimal(conf).quantize(Decimal("0.01"), rounding="ROUND_FLOOR")
            stereotype = nodes.get(node_id)['label']
            print(f'{node_id} {origin_label} {conf} {stereotype} {origin_label == "1"} {conf > 0}')
        elif origin_label == '1':
            conf = 0.0
            stereotype = nodes.get(node_id)['label']
            print(f'{node_id} {origin_label} {conf} {stereotype} {origin_label == "1"} {conf > 0}')


def graph_match(step, patterns):
    G1s = load_targets('my_mylyn', step)
    G2s = load_patterns(patterns)
    print('G1s', len(G1s), 'G2s', len(G2s))
    result_1, result_3, result_5, result_full = [], [], [], []
    for G1 in G1s[43:44]:

        print(G1.nodes(data=True))
        print(G1.edges(data=True))
        for node in G1.nodes(data=True):
            if node[1]['label'] == 'FIELD' and G1.has_edge(1, node[0]):
                print(f'1 号节点 declares 了一个 {node[0]} 号field')
            if node[1]['label'] == 'SET' and G1.has_edge(1, node[0]):
                print(f'1 号节点 declares 了一个 {node[0]} 号 SET')
            if node[1]['label'] == 'FIELD' and G1.has_edge(4, node[0]):
                print(f'4 号节点 declares 了一个 {node[0]} 号field')
            if node[1]['label'] == 'SET' and G1.has_edge(4, node[0]):
                print(f'4 号节点 declares 了一个 {node[0]} 号 SET')
        # for G1 in G1s[batch_index * 200: (batch_index + 1) * 200]:
        # if G1s.index(G1) in [290]:
        #     continue
        print(f'handling: {G1s.index(G1)}-{G1}')
        begin_time = time.time()
        total_match = 0
        confidence = dict()
        flag = False
        for G2 in G2s:
            curr_time = time.time()
            if curr_time - begin_time > 60 * 5:
                flag = True
                break
            GM = isomorphism.DiGraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
            if GM.subgraph_is_isomorphic():
                t = 0
                for sub_iter in GM.subgraph_isomorphisms_iter():
                    nodes = list(map(int, list(sub_iter.keys())))
                    ground_truth = 0
                    for node in nodes:
                        ground_truth += int(G1.nodes.get(node)['origin'])
                    # 匹配的子图中不属于 ground truth的节点数不能超过当前的预测补偿 step
                    if len(nodes) - ground_truth > step:
                        continue
                    total_match += 1
                    t += 1
                    for node in nodes:
                        if confidence.get(node):
                            confidence[node] = confidence[node] + 1
                        else:
                            confidence[node] = 1
                    # sub_G1: nx.DiGraph = G1.subgraph(nodes)
                    # print(sub_G1.nodes.data(), sub_G1.edges.data()
                print(G2.nodes(data=True), G2.edges(data=True), t)
        if flag: # 计算已经匹配出来的
            print(f'break {G1s.index(G1)}')
            break # 如果存在超时，直接跳过该 model
        print('total_match: ', total_match)
        confidence = dict(sorted(confidence.items(), key=lambda d: d[1], reverse=True))
        print(confidence)
        for i in confidence:
            confidence[i] = confidence.get(i) / total_match
        confidence = sorted(confidence.items(), key=lambda d: d[1], reverse=True)  # [(3, 1.0), (17, 0.5), (14, 0.5)]
        print(confidence)
        k = 0
        if k == 0:
            output, labels = [], []
            # print(confidence)
            for top_c in confidence:
                node_id = top_c[0]
                output.append(top_c[1])
                # print(G1.nodes.get(node_id))
                labels.append(int(G1.nodes.get(node_id)['origin']))
            true_number = 0
            for n in list(G1.nodes):
                true_number += int(G1.nodes.get(n)['origin'])
            if true_number > 0:
                result_full.append(calculate_result_full(labels, output, true_number))
                save_result_stereotype(confidence, G1.nodes)
    # print_result(result_1, 1)
    # print_result(result_3, 3)
    # print_result(result_5, 5)
    # pd.to_pickle(result_full, f'./origin_result/result_full_{step}_{batch_index}.pkl')
    print_result(result_full, 0)


def graph_build_and_gspan(min_sup, node_num, project_model_name='my_mylyn'):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    graph_index = 0
    with open('./no_graph.data', 'w') as f:
        # 读取code context model
        model_dir_list = get_models_by_ratio(project_model_name, 0, 0.8)
        print(len(model_dir_list))
        for model_dir in model_dir_list:
            # print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            model_file = join(model_path, 'code_context_model.xml')
            # 如果不存在模型，跳过处理
            if not os.path.exists(model_file):
                continue
            # 读取code context model,以及doxygen的结果
            tree = ET.parse(model_file)  # 拿到xml树
            code_context_model = tree.getroot()
            graphs = code_context_model.findall("graph")
            graph_text = f't # {graph_index}\n'
            f.write(graph_text)
            curr_index = 0
            for graph in graphs:
                vertices = graph.find('vertices')
                vertex_list = vertices.findall('vertex')
                vs = []
                for vertex in vertex_list:
                    stereotype, _id = vertex.get('stereotype'), int(vertex.get('id'))
                    # 去除 notfound
                    if not stereotype == 'NOTFOUND':
                        vs.append((_id, stereotype))
                for v in sorted(vs, key=lambda x: x[0]):
                    vertex_text = f'v {v[0] + curr_index} {v[1]}\n'
                    f.write(vertex_text)
                edges = graph.find('edges')
                edge_list = edges.findall('edge')
                for edge in edge_list:
                    start, end, label = int(edge.get('start')), int(edge.get('end')), edge.get('label')
                    keys = [x[0] for x in vs]
                    if start in keys and end in keys:
                        edge_text = f'e {start + curr_index} {end + curr_index} {label}\n'
                        f.write(edge_text)
                curr_index += len(vertex_list)
            graph_index += 1
        f.write('t # -1')
        f.close()
    print(graph_index)

    min_support = math.ceil(min_sup * graph_index)  # 0.02 * num_of_graphs
    print('min_support: ', min_support)
    # args_str = f'-h'
    if node_num == 0:
        args_str = f'-s {min_support} -d True ./no_graph.data'
    else:
        args_str = f'-s {min_support} -l {node_num} -u {node_num} -d True ./no_graph.data'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    main(FLAGS)


if __name__ == '__main__':
    # print(sys.argv)
    step = int(sys.argv[1]) if len(sys.argv) > 2 else 1
    # batch_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0 # 798 / 200 = 5 0,1,2,3
    # print(step, batch_index)
    min_sup = 0.015
    node_num = 0
    # 挖掘模式库 这里的 gsan库有问题，需要根据报错，将包源码的 append 方法修改为 _append 即可
    # graph_build_and_gspan(min_sup=min_sup, node_num=node_num)
    graph_match(step=step, patterns=f'./origin_patterns/no-sup-{min_sup}')
