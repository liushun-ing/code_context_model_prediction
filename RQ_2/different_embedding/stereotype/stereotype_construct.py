"""
用于构建 GNN 的输入
格式如下：
节点信息(nodes.tsv)：(node_id, code_embedding) node_id即java_code的文件id即可，code_embedding 即 200 embedding vector
边信息(edges.tsv)：(start_node_id, end_node_id, relation_vector) relation_vector即四种关系的one-hot编码表示
[declares, calls, inherits, implements]
"""

import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from dataset_split_util import get_models_by_ratio

root_path = join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
                 'params_validation',
                 'git_repo_code')

edge_vector = {
    "declares": 0,
    "calls": 1,
    "inherits": 2,
    "implements": 3
}
dataset_ratio = {
    'train': [0.0, 0.8],
    'valid': [0.8, 0.9],
    'test': [0.9, 1.0]
}

step_1 = ['NOTFOUND', 'PREDICATE-COLLABORATOR', 'BOUNDARY-DATA_PROVIDER', 'LAZY_CLASS', 'CONSTRUCTOR-CONTROLLER',
          'CONSTRUCTOR-LOCAL_CONTROLLER', 'COMMANDER', 'INCIDENTAL', 'PREDICATE', 'EMPTY', 'ABSTRACT',
          'SET-COLLABORATOR', 'GET', 'SET-LOCAL_CONTROLLER', 'COMMAND-COLLABORATOR', 'MINIMAL_ENTITY',
          'NON_VOID_COMMAND-COLLABORATOR', 'CONTROLLER', 'DEGENERATE', 'DATA_CLASS', 'PROPERTY-COLLABORATOR',
          'NON_VOID_COMMAND-LOCAL_CONTROLLER', 'PROPERTY-LOCAL_CONTROLLER', 'VOID_ACCESSOR-COLLABORATOR', 'POOL',
          'FACTORY-COLLABORATOR', 'FIELD', 'BOUNDARY-COMMANDER', 'BOUNDARY', 'PREDICATE-LOCAL_CONTROLLER', 'ENTITY',
          'GET-LOCAL_CONTROLLER', 'LOCAL_CONTROLLER', 'FACTORY', 'SET', 'INTERFACE', 'CONSTRUCTOR-COLLABORATOR',
          'COLLABORATOR', 'FACTORY-LOCAL_CONTROLLER', 'DATA_PROVIDER', 'CONSTRUCTOR', 'PROPERTY', 'NON_VOID_COMMAND',
          'COMMAND', 'OTHER', 'GET-COLLABORATOR', 'COMMAND-LOCAL_CONTROLLER']

onehot_1 = np.eye(len(step_1) + 1)  # 48

step_2 = ['CONSTRUCTOR', 'FACTORY-CONTROLLER', 'FACTORY-COLLABORATOR', 'GET-COLLABORATOR', 'BOUNDARY',
          'COMMAND-LOCAL_CONTROLLER', 'PREDICATE-COLLABORATOR', 'DATA_PROVIDER', 'ENTITY', 'COMMANDER',
          'LOCAL_CONTROLLER', 'FACTORY-LOCAL_CONTROLLER', 'CONSTRUCTOR-LOCAL_CONTROLLER', 'LAZY_CLASS', 'INCIDENTAL',
          'POOL', 'DEGENERATE', 'GET', 'SET-COLLABORATOR', 'EMPTY', 'BOUNDARY-DATA_PROVIDER', 'MINIMAL_ENTITY',
          'NON_VOID_COMMAND-COLLABORATOR', 'FACTORY', 'VOID_ACCESSOR-COLLABORATOR', 'FIELD', 'INTERFACE', 'PROPERTY',
          'BOUNDARY-COMMANDER', 'PREDICATE-LOCAL_CONTROLLER', 'SET', 'NON_VOID_COMMAND', 'PROPERTY-COLLABORATOR',
          'GET-LOCAL_CONTROLLER', 'OTHER', 'NOTFOUND', 'ABSTRACT', 'DATA_CLASS', 'CONSTRUCTOR-CONTROLLER',
          'NON_VOID_COMMAND-LOCAL_CONTROLLER', 'COLLABORATOR', 'CONSTRUCTOR-COLLABORATOR', 'PROPERTY-LOCAL_CONTROLLER',
          'PREDICATE', 'COMMAND', 'CONTROLLER', 'SET-LOCAL_CONTROLLER', 'COMMAND-COLLABORATOR']
onehot_2 = np.eye(len(step_2) + 1)  # 49

step_3 = ['SET-COLLABORATOR', 'NON_VOID_COMMAND-COLLABORATOR', 'CONSTRUCTOR-CONTROLLER', 'PREDICATE', 'CONSTRUCTOR',
          'PROPERTY-COLLABORATOR', 'VOID_ACCESSOR-COLLABORATOR', 'GET-COLLABORATOR', 'DATA_CLASS', 'LAZY_CLASS',
          'CONSTRUCTOR-COLLABORATOR', 'NON_VOID_COMMAND', 'FIELD', 'PROPERTY', 'COMMAND-LOCAL_CONTROLLER',
          'LOCAL_CONTROLLER', 'ENTITY', 'NOTFOUND', 'SET-LOCAL_CONTROLLER', 'DEGENERATE', 'GET',
          'PREDICATE-COLLABORATOR', 'EMPTY', 'BOUNDARY', 'MINIMAL_ENTITY', 'FACTORY', 'ABSTRACT', 'POOL', 'COMMANDER',
          'INCIDENTAL', 'BOUNDARY-DATA_PROVIDER', 'FACTORY-COLLABORATOR', 'CONSTRUCTOR-LOCAL_CONTROLLER',
          'GET-LOCAL_CONTROLLER', 'DATA_PROVIDER', 'INTERFACE', 'NON_VOID_COMMAND-LOCAL_CONTROLLER', 'SET',
          'FACTORY-LOCAL_CONTROLLER', 'COLLABORATOR', 'PREDICATE-LOCAL_CONTROLLER', 'COMMAND-COLLABORATOR',
          'BOUNDARY-COMMANDER', 'CONTROLLER', 'PROPERTY-LOCAL_CONTROLLER', 'OTHER', 'COMMAND']
onehot_3 = np.eye(len(step_3) + 1)  # 48


def save_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name):
    if step == 1:
        onehot = onehot_1
        step_s = step_1
    elif step == 2:
        onehot = onehot_2
        step_s = step_2
    else:
        onehot = onehot_3
        step_s = step_3
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'{str(step)}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        base = 0
        for graph in graphs:
            # 创建保存nodes和edges的数据结构
            nodes_tsv = list()
            edges_tsv = list()
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                stereotype = vertex.get('stereotype')
                _id = int(vertex.get('id'))
                # 去找 embedding 并添加  这儿不知道为什么会多一层列表
                embedding = onehot[step_s.index(stereotype)] if stereotype in step_s else onehot[len(step_s)]
                nodes_tsv.append([_id, embedding.tolist(), int(vertex.get('origin'))])
            edges = graph.find('edges')
            edge_list = edges.findall('edge')
            for edge in edge_list:
                edges_tsv.append([int(edge.get('start')), int(edge.get('end')), edge_vector[edge.get('label')]])
            dest = join(dest_path, f'model_dataset_{str(step)}', dataset,
                        f'{project_model_name}_{model_dir}_{str(base)}')
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(dest)
            # 如果没有节点，或者没有边，或者没有正边，都需要过滤掉
            if len(nodes_tsv) > 0 and len(edges_tsv) > 0:
                pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding', 'label']).to_csv(
                    join(dest, 'nodes.tsv'), index=False)
                pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation']).to_csv(
                    join(dest, 'edges.tsv'), index=False)
            base += 1


def main_func(description: str, step: int, dest_path: str):
    """
    构建并拆分数据集
    all: 四个项目分别拆分成train,valid,test : 比例8:1:1 \n
    onlymylyn: mylyn作为train, 其余三个项目各自拆分为valid,test=3：7
    nopde:pde拆分为valid:test=3:7,其余为train
    noplatform: platform差分为valid:test=3:7,其余为train

    :param description: all, onlymylyn, nopde, noplatform
    :param step: 步长
    :param dest_path: 数据集保存的路径
    :return:
    """
    if os.path.exists(join(dest_path, f'model_dataset_{str(step)}')):
        shutil.rmtree(join(dest_path, f'model_dataset_{str(step)}'))
    if description == 'all':
        project_model_list = ['my_mylyn', 'my_pde', 'my_platform']
        for project_model_name in project_model_list:
            project_path = join(root_path, project_model_name, 'repo_first_3')
            for dataset in dataset_ratio:
                ratios = dataset_ratio.get(dataset)
                model_dir_list = get_models_by_ratio(project_model_name, ratios[0], ratios[1])
                model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
                save_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name)
