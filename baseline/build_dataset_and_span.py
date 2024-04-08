import math
import os
from os.path import join
from gspan_mining.config import parser
from gspan_mining.main import main
import xml.etree.ElementTree as ET

from dataset_split_util import get_models_by_ratio

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    graph_index = 0
    with open('./graph.data', 'w') as f:
        # 读取code context model
        model_dir_list = get_models_by_ratio('my_mylyn', 0, 0.84)
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
            for graph in graphs:
                graph_text = f't # {graph_index}\n'
                f.write(graph_text)
                vertices = graph.find('vertices')
                vertex_list = vertices.findall('vertex')
                vs = []
                for vertex in vertex_list:
                    stereotype, _id = vertex.get('stereotype'), vertex.get('id')
                    vs.append((_id, stereotype))
                for v in sorted(vs, key=lambda x: int(x[0])):
                    vertex_text = f'v {v[0]} {v[1]}\n'
                    f.write(vertex_text)
                edges = graph.find('edges')
                edge_list = edges.findall('edge')
                for edge in edge_list:
                    start, end, label = edge.get('start'), edge.get('end'), edge.get('label')
                    edge_text = f'e {start} {end} {label}\n'
                    f.write(edge_text)
                graph_index += 1
        f.write('t # -1')
        f.close()
    print(graph_index)

    min_support = math.ceil(0.007 * (graph_index - 1))  # 0.02 * num_of_graphs
    print('min_support: ', min_support)
    args_str = f'-s {min_support} ./graph.data'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = main(FLAGS)

    # with open('./patterns', 'w+') as res:
    #     res.write(gs.graphs.items())
    #     res.close()
    # print()
    # print(gs)


main_func('my_mylyn')
