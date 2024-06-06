import os
from collections import Counter
from os.path import join
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np

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
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=node.get('origin'))
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
    count = 0
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
        for i in range(len(graphs)):
            print(count, model_dir)
            count = count + 1
        G1s = G1s + get_graph(graphs, step)
    return G1s


def read_result(step: int):
    result = []
    with open(f'./origin_result/no_new_match_result_{step}_1.txt', 'r') as f:
        lines = f.readlines()
        curr = []
        for line in lines:
            line = line.strip()
            if line.startswith('---') or line.startswith('node_id'):
                if len(curr) > 0:
                    result.append(curr.copy())
                    curr = []
                continue
            else:
                curr.append(line.split(' '))
    if len(curr) > 0:
        result.append(curr)
    return result


def find_connected_nodes_with_labels(digraph, node_id):
    reachable_from_node, reachable_to_node = [], []
    from_edges, to_edges = [], []
    # 找到所有从 node_id 可达的节点
    for edge in digraph.edges():
        if edge[0] == node_id:
            reachable_from_node.append(edge[1])
            from_edges.append(digraph.get_edge_data(edge[0], edge[1])['label'])
    # 找到所有可以到达 node_id 的节点
    for edge in digraph.edges():
        if edge[1] == node_id:
            reachable_to_node.append(edge[0])
            to_edges.append(digraph.get_edge_data(edge[0], edge[1])['label'])
    # 获取这些节点的 label 属性
    reachable_from_node_labels = [digraph.nodes[n]['label'] for n in reachable_from_node]
    reachable_to_node_labels = [digraph.nodes[n]['label'] for n in reachable_to_node]
    print(node_id, reachable_from_node, reachable_from_node_labels, from_edges, reachable_to_node, reachable_to_node_labels, to_edges)

    return reachable_from_node_labels, reachable_to_node_labels


def field_analyse(step):
    G1s = load_targets('my_mylyn', step)
    print('G1s', len(G1s))
    result = read_result(step)
    print(len(result))
    # 用于存储所有图的 label 分布
    all_labels = [[], []]
    count = 0
    for curr in result:
        print(count)
        count = count + 1
        for node in curr:
            if node[3] == 'FIELD' and node[4] == 'True' and node[5] == 'False':
                # print(node)
                index = result.index(curr)
                g = G1s[index]
                node_id = int(node[0])
                from_labels, to_labels = find_connected_nodes_with_labels(g, node_id)
                all_labels[0] = all_labels[0] + from_labels
                all_labels[1] = all_labels[1] + to_labels
    frequency_counter = Counter(all_labels[0])
    total_count = len(all_labels[0])
    # 计算频率和占比
    frequencies_and_ratios = {string: (count, count / total_count) for string, count in frequency_counter.items()}
    sorted_labels = sorted(frequencies_and_ratios.items(), key=lambda item: item[1][0], reverse=True)
    for string, (frequency, ratio) in sorted_labels:
        print(f"{string} {frequency} {ratio:.2%}")
    print('-------------------------------------')
    frequency_counter = Counter(all_labels[1])
    total_count = len(all_labels[1])
    # 计算频率和占比
    frequencies_and_ratios = {string: (count, count / total_count) for string, count in frequency_counter.items()}
    sorted_labels = sorted(frequencies_and_ratios.items(), key=lambda item: item[1][0], reverse=True)
    for string, (frequency, ratio) in sorted_labels:
        print(f"{string} {frequency} {ratio:.2%}")
    print('-----------------all--------------------')
    frequency_counter = Counter(all_labels[0] + all_labels[1])
    total_count = len(all_labels[0] + all_labels[1])
    # 计算频率和占比
    frequencies_and_ratios = {string: (count, count / total_count) for string, count in frequency_counter.items()}
    sorted_labels = sorted(frequencies_and_ratios.items(), key=lambda item: item[1][0], reverse=True)
    for string, (frequency, ratio) in sorted_labels:
        print(f"{string} {frequency} {ratio:.2%}")


if __name__ == '__main__':
    field_analyse(step=1)
