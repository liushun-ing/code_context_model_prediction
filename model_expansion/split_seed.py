"""
将 code context model 拆分 seed
"""
import os
from os.path import join
import xml.etree.ElementTree as ET


root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def exist_edge(new_vertices: list[ET.Element], new_edge: ET.Element):
    start = new_edge.get('start')
    end = new_edge.get('end')
    count = 0
    for node in new_vertices:
        if node.get('id') == start or node.get('id') == end:
            count += 1
    return count == 2


def get_all_seed_graph(vertices: list[ET.Element], edges: list[ET.Element], step: int):
    max_sup = len(vertices) - step
    vertices = sorted(vertices, key=lambda x: x.get('timestamp'))
    res = []
    for sup in range(1, max_sup + 1):
        new_vertices = vertices[0: sup]
        new_edges = []
        for new_edge in edges:
            if exist_edge(new_vertices, new_edge):
                new_edges.append(new_edge)
        res.append((new_vertices, new_edges))
    return res


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # index = model_dir_list.index('5232')
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # shutil.copy(rename_model_file, model_file)
        # 下面对code context model进行component拆分
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        graphs = root.findall("graph")
        steps = [1]
        for step in steps:
            total = 0
            new_root = ET.Element("code_context_model")
            new_root.set('commit', root.get('commit'))
            new_root.set('first_time', root.get('first_time'))
            new_root.set('last_time', root.get('last_time'))
            for graph in graphs:
                # 根据时间顺序进行种子选举
                repo_name = graph.get('repo_name')
                repo_path = graph.get('repo_path')
                vertices = graph.find('vertices').findall('vertex')
                edges = graph.find('edges').findall('edge')
                all_seed_graph = get_all_seed_graph(vertices, edges, step)
                for seed_vertices, seed_edges in all_seed_graph:
                    # 可能生成多个图
                    new_graph = ET.SubElement(new_root, 'graph')
                    new_graph.set('repo_name', repo_name)
                    new_graph.set('repo_path', repo_path)
                    new_vertices = ET.SubElement(new_graph, 'vertices')
                    new_edges = ET.SubElement(new_graph, 'edges')
                    new_vertices.set('total', str(len(seed_vertices)))
                    new_edges.set('total', str(len(seed_edges)))
                    id_map = dict()
                    count = 0
                    for seed_node in seed_vertices:
                        id_map[seed_node.get('id')] = str(count)
                        count += 1
                    for node_element in seed_vertices:
                        new_vertex = ET.SubElement(new_vertices, 'vertex')
                        new_vertex.set('id', id_map.get(node_element.get('id')))
                        new_vertex.set('ref_id', node_element.get('ref_id'))
                        new_vertex.set('kind', node_element.get('kind'))
                        new_vertex.set('label', node_element.get('label'))
                        new_vertex.set('timestamp', node_element.get('timestamp'))
                        new_vertex.set('seed', '1')
                    for edge_element in seed_edges:
                        new_edge = ET.SubElement(new_edges, 'edge')
                        new_edge.set('start', id_map.get(edge_element.get('start')))
                        new_edge.set('end', id_map.get(edge_element.get('end')))
                        new_edge.set('label', edge_element.get('label'))
                    total += 1
            new_root.set('total', str(total))
            new_tree = ET.ElementTree(new_root)
            # 将XML写入文件
            new_tree.write(join(model_path, f'{step}_step_seed_model.xml'))



# ecf
# main_func('my_ecf')
# pde
main_func('my_pde')
# platform
main_func('my_platform')
# # mylyn
# print(root_path)
main_func('my_mylyn')
