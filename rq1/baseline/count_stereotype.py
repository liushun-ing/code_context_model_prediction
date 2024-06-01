import os
from collections import Counter
from os.path import join
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
from networkx import DiGraph

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
        # first_time = code_context_model.get('last_time')
        first_time = code_context_model.get('first_time')
        all_models.append([model_dir, first_time])
    all_models = sorted(all_models, key=lambda x: x[1])
    m = np.array(all_models)
    return m[int(len(m) * start_ratio):int(len(m) * end_ratio), 0]


def get_graph(graphs: list[ET.Element]):
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
        for node in vertex_list:
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=node.get('origin'),
                       seed=node.get('origin'))
            # if int(node.get('origin')) == 1:
            true_node += 1
        for link in edge_list:
            g.add_edge(int(link.get('start')), int(link.get('end')), label=link.get('label'))
            # if int(link.get('origin')) == 1:
            #     true_edge += 1
        if true_node > 1:
            gs.append(g)
    return gs


def load_targets(project_model_name: str, step, mode, start, end):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    G1s = []
    # 读取code context model
    model_dir_list = get_models_by_ratio(project_model_name, start, end)
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # print(model_dir_list)
    for model_dir in model_dir_list:
        # print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        if mode == 'origin':
            model_file = join(model_path, f'code_context_model.xml')
        elif mode == 'expand':
            model_file = join(model_path, f'new_{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        tree = ET.parse(model_file)  # 拿到xml树
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        G1s = G1s + get_graph(graphs)
    return G1s


def count_stereotype(step, mode, start, end):
    G1s = load_targets('my_mylyn', step, mode, start, end)
    print('G1s', len(G1s))
    # 用于存储所有图的 label 分布
    all_label_values = []

    # 遍历每个图，获取每个图的 label 数据并合并到一起
    for g in G1s:
        labels = nx.get_node_attributes(g, 'label')
        label_values = list(labels.values())
        all_label_values.extend(label_values)

    # 使用 Counter 统计所有图的 label 分布
    label_distribution = Counter(all_label_values)

    # 计算所有图中节点的总数
    total_nodes = len(all_label_values)

    # 输出每个图的 label 分布数量和比例
    print("Label Distribution across all graphs:")
    sorted_labels = sorted(label_distribution.items(), key=lambda item: item[1], reverse=True)

    for label, frequency in sorted_labels:
        ratio = frequency / total_nodes
        print(f'{label} {frequency} {ratio:.2%}')
    return G1s


def analyse_field(G1s: list[DiGraph]):
    # 用于存储结果的全局数组
    results = []
    # 遍历每个图
    for graph in G1s:
        # 遍历每条边
        for src, dst in graph.edges():
            # 获取src节点和dst节点的stereotype属性
            src_stereotype = graph.nodes[src].get('label')
            dst_stereotype = graph.nodes[dst].get('label')
            # 获取边的label属性
            edge_label = graph.edges[src, dst].get('label')
            # 如果src节点的stereotype是FIELD
            if src_stereotype == 'FIELD':
                results.append((edge_label, dst_stereotype))
            # 如果dst节点的stereotype是FIELD
            if dst_stereotype == 'FIELD':
                results.append((edge_label, src_stereotype))
    # 使用Counter统计频率
    label_counter = Counter([item[0] for item in results])
    stereotype_counter = Counter([item[1] for item in results])
    # 计算总数
    total_count = len(results)
    # 计算并打印结果
    print("kind frequency and percentage:")
    for label, freq in label_counter.items():
        percentage = (freq / total_count) * 100
        print(f"{label} {freq} {percentage:.2f}%")
    print("\nStereotype frequency and percentage:")
    sorted_labels = sorted(stereotype_counter.items(), key=lambda item: item[1], reverse=True)
    for stereotype, freq in sorted_labels:
        percentage = (freq / total_count) * 100
        print(f"{stereotype} {freq} {percentage:.2f}%")


if __name__ == '__main__':
    mode = 'expand'
    # mode = 'origin'
    step = 1
    start = 0.9
    end = 1
    G1s = count_stereotype(step=step, mode=mode, start=start, end=end)
    print('-------------------------')
    analyse_field(G1s)

