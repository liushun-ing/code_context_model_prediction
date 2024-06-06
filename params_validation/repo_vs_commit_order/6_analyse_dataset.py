"""
用于统计context model的节点和边的类型分布
"""
import os
import statistics

import networkx as nx
from os.path import join
import xml.etree.ElementTree as ET

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')


def calculate_type_and_edge(graphs: list[ET.Element]):
    types_count = {
        'class': 0,
        'interface': 0,
        'function': 0,
        'variable': 0
    }
    edges_count = {
        'declares': 0,
        'calls': 0,
        'uses': 0,
        'inherits': 0,
        'implements': 0
    }
    for graph in graphs:
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        edges = graph.find('edges')
        edge_list = edges.findall('edge')
        for vertex in vertex_list:
            kind = vertex.get('kind')
            types_count[kind] = types_count[kind] + 1
        for edge in edge_list:
            kind = edge.get('label')
            start = int(edge.get('start'))
            end = int(edge.get('end'))
            if kind == 'calls':
                for node in vertex_list:
                    if int(node.get('id')) == end and node.get('kind') == 'variable':
                        kind = 'uses'
                        break
            edges_count[kind] = edges_count[kind] + 1
    return types_count, edges_count


def calculate_stats(data):
    stats = {}
    for key, values in data.items():
        if values:
            stats[key] = {
                'Min': min(values),
                'Max': max(values),
                'Mean': statistics.mean(values),
                'Median': statistics.median(values),
                'SD': statistics.stdev(values) if len(values) > 1 else 0,
                'Sum': sum(values)
            }
        else:
            stats[key] = {
                'Min': None,
                'Max': None,
                'Mean': None,
                'Median': None,
                'SD': None,
                'Sum': None
            }
    metrics = ['Min', 'Max', 'Mean', 'Median', 'SD', 'Sum']
    headers = list(stats.keys())

    print(f"{'Metric':<10}", end="")
    for header in headers:
        print(f"{header:<10}", end="")
    print()

    for metric in metrics:
        print(f"& {metric:<8}", end="")
        for header in headers:
            value = stats[header][metric]
            if value is None:
                print(f"& {value:<8}", end="")
            elif isinstance(value, float):
                print(f"& {value:<8.2f}", end="")
            else:
                print(f"& {value:<8}", end="")
        print()


def main_func(project_name: str):
    print(project_name)
    project_path = join(root_path, project_name)
    strategy_dir_list = os.listdir(project_path)
    for strategy_dir in strategy_dir_list:
        strategy_path = join(project_path, strategy_dir)
        model_dir_list = os.listdir(strategy_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        model_total = 0
        types_count = {
            'class': [],
            'interface': [],
            'function': [],
            'variable': []
        }
        edges_count = {
            'declares': [],
            'calls': [],
            'uses': [],
            'inherits': [],
            'implements': []
        }
        for model_dir in model_dir_list:
            model_path = join(strategy_path, model_dir, 'code_context_model.xml')
            if not os.path.exists(model_path):
                continue
            model_total += 1
            tree = ET.parse(model_path)  # 拿到xml树
            # 获取XML文档的根元素
            root = tree.getroot()
            single_type, single_edge = calculate_type_and_edge(root.findall("graph"))
            for key, value in single_type.items():
                types_count[key].append(value)
            for key, value in single_edge.items():
                edges_count[key].append(value)
        calculate_stats(types_count)
        calculate_stats(edges_count)


if __name__ == '__main__':
    # main_func('my_pde')
    # main_func('my_platform')
    main_func('my_mylyn')
