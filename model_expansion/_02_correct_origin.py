"""
由于进行了component拆分，所有可能扩展的时候会引入true节点，所以需要重新修正
"""
import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def load_origin_model(model_path):
    tree = ET.parse(model_path)  # 拿到xml树
    # 获取XML文档的根元素
    root = tree.getroot()
    graphs = root.findall("graph")
    all_nodes = []
    all_edges = []
    for graph in graphs:
        vertices = graph.find('vertices').findall('vertex')
        edges = graph.find('edges').findall('edge')
        all_nodes = all_nodes + vertices
        all_edges = all_edges + edges
    return all_nodes, all_edges


def exist_node(vertices: list[ET.Element], ref_id):
    for node in vertices:
        if node.get('ref_id') == ref_id:
            return True
    return False


def exist_edge(all_nodes: list[ET.Element], nodes: list[ET.Element], start, end):
    s = False
    for node in nodes:
        if node.get('id') == start:
            if exist_node(all_nodes, node.get('ref_id')):
                s = True
            break
    if not s:
        return False
    e = False
    for node in nodes:
        if node.get('id') == end:
            if exist_node(all_nodes, node.get('ref_id')):
                e = True
            break
    return e


def main_func(project_model_name: str, step):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # index = model_dir_list.index('5232')
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'new1_{step}_step_expanded_model.xml')
        rename_model_file = join(model_path, '_code_context_model.xml')
        all_nodes, all_edges = load_origin_model(rename_model_file)
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # shutil.copy(rename_model_file, model_file)
        # 下面对code context model进行component拆分
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        graphs = root.findall("graph")
        for graph in graphs:
            print('graph------------{}+{}'.format(len(all_nodes), len(all_edges)))
            vertices = graph.find('vertices').findall('vertex')
            edges = graph.find('edges').findall('edge')
            for node in vertices:
                if node.get('origin') == '0' and exist_node(all_nodes, node.get('ref_id')):
                    node.set('origin', '1')
                    print('changed node origin to 1')
            for link in edges:
                if link.get('origin') == '0' and exist_edge(all_nodes, vertices, link.get('start'), link.get('end')):
                    link.set('origin', '1')
                    # print('changed edge origin to 1')
        # 将XML写入文件
        tree.write(model_file)


if __name__ == '__main__':
    # ecf
    # main_func('my_ecf')
    # pde
    # main_func('my_pde')
    # platform
    # main_func('my_platform')
    # # mylyn
    # print(root_path)
    main_func('my_mylyn', 1)
    main_func('my_mylyn', 2)
    main_func('my_mylyn', 3)
