"""
扩展code context model的工具类，提供加载模型和保存模型方法
"""
import os
import xml.etree.ElementTree as ET

from xmlparser.doxygen_main.Graph import Graph
from xmlparser.doxygen_main.expand_graph import function_like_kind, variable_like_kind


def load_code_context_model(model_file: str):
    """
    加载code context model为 Graph 对象

    :param model_file: context model xml文件路径
    :return: 加载的图集合
    """
    tree = ET.parse(model_file)  # 拿到xml树
    # 获取XML文档的根元素
    code_context_model = tree.getroot()
    graphs = code_context_model.findall("graph")
    graph_list: list[Graph] = []
    for graph in graphs:
        g = Graph()
        g.set_repo_name(graph.get('repo_name'))
        g.set_repo_path(graph.get('repo_path'))
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        for vertex in vertex_list:
            g.add_vertex_origin(vertex.get('ref_id'), vertex.get('kind'), vertex.get('label'))
        edges = graph.find('edges')
        edge_list = edges.findall('edge')
        for edge in edge_list:
            g.add_edge_origin(int(edge.get('start')), int(edge.get('end')), edge.get('label'))
        graph_list.append(g)
    print('load {} code context model over~~~~~~~~~~~~'.format(model_file))
    return graph_list


def save_expanded_model(graph_list: list[Graph], save_path: str):
    """
    保存扩展的模型到文件

    :param graph_list: 图集合
    :param save_path: 保存的文件路径
    :return: 无
    """
    if os.path.isfile(save_path):
        os.remove(save_path)
    # if len(graph_list) == 0:
    #     return
    # 写图文件,将几个图组合在一起，就是代码上下文模型
    model_root = ET.Element("expanded_model")
    model_root.set('total', str(len(graph_list)))
    for graph in graph_list:
        graph_node = ET.SubElement(model_root, 'graph')
        graph_node.set('repo_name', graph.repo_name)
        graph_node.set('repo_path', graph.repo_path)
        vertices = ET.SubElement(graph_node, 'vertices')
        vertices.set('total', str(len(graph.vertices)))
        for vertex in graph.vertices:
            v_node = ET.SubElement(vertices, 'vertex')
            v_node.set('id', str(vertex.id))
            v_node.set('ref_id', vertex.ref_id)
            v_node.set('kind', vertex.kind)
            v_node.set('label', vertex.label)
            v_node.set('origin', str(vertex.origin))
            if vertex.kind in function_like_kind or vertex.kind in variable_like_kind:
                # vertex.location.print()
                v_node.set('file', str(vertex.location.file))
                v_node.set('line', str(vertex.location.line))
                v_node.set('column', str(vertex.location.column))
                v_node.set('body_file', str(vertex.location.body_file))  # 这个可能为None，所以也转一下
                v_node.set('body_start', str(vertex.location.body_start))
                v_node.set('body_end', str(vertex.location.body_end))
        edges = ET.SubElement(graph_node, 'edges')
        edges.set('total', str(len(graph.edges)))
        for edge in graph.edges:
            e_node = ET.SubElement(edges, 'edge')
            e_node.set('start', str(edge.start))
            e_node.set('end', str(edge.end))
            e_node.set('label', edge.label)
            e_node.set('origin', str(edge.origin))
    model_tree = ET.ElementTree(model_root)
    model_tree.write(save_path)
    print('~~~~~~expanded model saved in {0}~~~~~~'.format(save_path))
