"""
将code context model拆分component
"""
import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

import networkx as nx

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def get_graph(vertices: list[ET.Element], edges: list[ET.Element]):
    g = nx.DiGraph()
    # 转化为图结构
    for node in vertices:
        g.add_node(node.get('id'))
    for link in edges:
        g.add_edge(link.get('start'), link.get('end'))
    return g


def get_node(vertices: list[ET.Element], _id):
    for node in vertices:
        if node.get('id') == _id:
            return node


def get_edge(edges: list[ET.Element], start, end):
    for edge in edges:
        if edge.get('start') == start and edge.get('end') == end:
            return edge


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # index = model_dir_list.index('4585')
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        # rename_model_file = join(model_path, '_code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # shutil.copy(rename_model_file, model_file)
        # 下面对code context model进行component拆分
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        graphs = root.findall("graph")
        new_root = ET.Element("code_context_model")
        new_root.set('commit', root.get('commit'))
        total = 0
        new_root.set('first_time', root.get('first_time'))
        new_root.set('last_time', root.get('last_time'))
        for graph in graphs:
            repo_name = graph.get('repo_name')
            repo_path = graph.get('repo_path')
            vertices = graph.find('vertices').findall('vertex')
            edges = graph.find('edges').findall('edge')
            net_graph = get_graph(vertices, edges)
            connected_components = list(nx.weakly_connected_components(net_graph))
            for cc in connected_components:
                subgraph = net_graph.subgraph(cc)
                if len(subgraph.nodes) <= 1:
                    continue
                new_graph = ET.SubElement(new_root, 'graph')
                new_graph.set('repo_name', repo_name)
                new_graph.set('repo_path', repo_path)
                new_vertices = ET.SubElement(new_graph, 'vertices')
                new_edges = ET.SubElement(new_graph, 'edges')
                new_vertices.set('total', str(len(subgraph.nodes)))
                new_edges.set('total', str(len(subgraph.edges)))
                id_map = dict()
                count = 0
                print(subgraph.nodes)
                for i in sorted(subgraph.nodes, key=lambda x: int(x)):
                    id_map[i] = str(count)
                    count += 1
                for node in subgraph.nodes:
                    node_element = get_node(vertices, node)
                    new_vertex = ET.SubElement(new_vertices, 'vertex')
                    new_vertex.set('id', id_map.get(node_element.get('id')))
                    new_vertex.set('ref_id', node_element.get('ref_id'))
                    new_vertex.set('kind', node_element.get('kind'))
                    new_vertex.set('label', node_element.get('label'))
                for edge in subgraph.edges:
                    edge_element = get_edge(edges, edge[0], edge[1])
                    new_edge = ET.SubElement(new_edges, 'edge')
                    new_edge.set('start', id_map.get(edge_element.get('start')))
                    new_edge.set('end', id_map.get(edge_element.get('end')))
                    new_edge.set('label', edge_element.get('label'))
                total += 1
        new_root.set('total', str(total))
        new_tree = ET.ElementTree(new_root)
        # 将XML写入文件
        new_tree.write(model_file)



# ecf
# main_func('my_ecf')
# pde
main_func('my_pde')
# platform
main_func('my_platform')
# mylyn
# print(root_path)
main_func('my_mylyn')
