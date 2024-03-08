"""
用于构建 GNN 的输入
主要包括保存所有 code context model 作为训练输入
格式如下：
节点信息(nodes.tsv)：(node_id, code_embedding) node_id即java_code的文件id即可，code_embedding 即 200 embedding vector
边信息(edges.tsv)：(start_node_id, end_node_id, relation_vector) relation_vector即四种关系的one-hot编码表示
[declares, calls, inherits, implements]
"""

import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

import pandas as pd

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')
gnn_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'GNN_Link_Prediction')

edge_vector = {
    "declares": 0,
    "calls": 1,
    "inherits": 2,
    "implements": 3
}


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name)
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list[0:500]:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, '1_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读 embedding
        embedding_list = pd.read_pickle(join(model_path, 'embedding.pkl'))
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        base = 0
        for graph in graphs:
            # 创建保存nodes和edges的数据结构
            nodes_tsv = list()
            edges_tsv = list()
            count = 0
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')])
                _id = int(vertex.get('id'))
                # 去找 embedding 并添加  这儿不知道为什么会多一层列表
                embedding = embedding_list[embedding_list['id'] == node_id]['embedding'].iloc[0]
                nodes_tsv.append([_id, embedding.tolist()])
            edges = graph.find('edges')
            edge_list = edges.findall('edge')
            for edge in edge_list:
                edges_tsv.append([int(edge.get('start')), int(edge.get('end')), edge_vector[edge.get('label')],
                                  int(edge.get('origin'))])
                if edge.get('origin') == '1':
                    count += 1
            dest = join(gnn_path, 'model_dataset_1', model_dir + '_' + str(base))
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(dest)
            # 如果没有节点，或者没有边，或者没有正边，都需要过滤掉
            if len(nodes_tsv) > 0 and len(edges_tsv) > 0 and count > 0:
                pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding']).to_csv(
                    join(dest, 'nodes.tsv'), index=False)
                pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation', 'label']).to_csv(
                    join(dest, 'edges.tsv'), index=False)
            base += 1


main_func('my_mylyn')
