"""
用于构建 GNN 的输入
格式如下：
节点信息(nodes.tsv)：(node_id, code_embedding) node_id即java_code的文件id即可，code_embedding 即 200 embedding vector
边信息(edges.tsv)：(start_node_id, end_node_id, relation_vector) relation_vector即四种关系的one-hot编码表示
[declares, calls, inherits, implements]
"""
import math
import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

import pandas as pd
import torch

from dataset_split_util import get_models_by_ratio

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')

edge_vector = {
    "declares": 0,
    "calls": 1,
    "inherits": 2,
    "implements": 3,
    "uses": 4,
    "declares_rev": 6,
    "calls_rev": 7,
    "inherits_rev": 8,
    "implements_rev": 9,
    "uses_rev": 10,
}
dataset_ratio = {
    'train': [0.0, 0.8],
    'valid': [0.8, 0.9],
    'test': [0.9, 1.0]
}
mylyn_dataset_ratio = {
    'train': [0, 0.8],
    'valid': [0.8, 0.9],
    'test': [0.9, 1]
}
SEED_EMBEDDING = torch.nn.Embedding(2, 256)

step_1 = ['POOL', 'PREDICATE-COLLABORATOR', 'DESTRUCTOR-LOCAL_CONTROLLER', 'PREDICATE', 'BOUNDARY',
          'CONSTRUCTOR-COLLABORATOR', 'FACTORY', 'PROPERTY-COLLABORATOR', 'NON_VOID_COMMAND-LOCAL_CONTROLLER',
          'PROPERTY', 'GET-LOCAL_CONTROLLER', 'PREDICATE-LOCAL_CONTROLLER', 'PURE_CONTROLLER', 'SET',
          'GET-COLLABORATOR', 'CONSTRUCTOR-LOCAL_CONTROLLER', 'LOCAL_CONTROLLER', 'CONSTRUCTOR', 'BOUNDARY-COMMANDER',
          'CONTROLLER', 'INTERFACE', 'COLLABORATOR', 'LAZY_CLASS', 'NON_VOID_COMMAND-COLLABORATOR', 'DEGENERATE',
          'INCIDENTAL', 'FIELD', 'VOID_ACCESSOR-COLLABORATOR', 'FACTORY-LOCAL_CONTROLLER', 'COMMAND', 'DATA_PROVIDER',
          'ABSTRACT', 'NOTFOUND', 'SET-LOCAL_CONTROLLER', 'BOUNDARY-DATA_PROVIDER', 'MINIMAL_ENTITY',
          'CONSTRUCTOR-CONTROLLER', 'ENTITY', 'OTHER', 'COMMANDER', 'NON_VOID_COMMAND', 'FACTORY-COLLABORATOR',
          'COMMAND-COLLABORATOR', 'DATA_CLASS', 'GET', 'EMPTY', 'FACTORY-CONTROLLER', 'LARGE_CLASS',
          'PROPERTY-LOCAL_CONTROLLER', 'SET-COLLABORATOR', 'COMMAND-LOCAL_CONTROLLER'
          ]
STEREOTYPE_EMBEDDING = torch.nn.Embedding(len(step_1), 256)


def adjust_edge(edge: ET.Element, nodes: list[ET.Element]):
    start = int(edge.get('start'))
    end = int(edge.get('end'))
    label = edge.get('label')
    if label == 'calls':
        for node in nodes:
            if int(node.get('id')) == end and node.get('kind') == 'variable':
                return [start, end, edge_vector['uses']]
        return [start, end, edge_vector[label]]
    else:
        return [start, end, edge_vector[label]]


def save_composed_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name, embedding_type,
                        description):
    if embedding_type == 'astnn+codebert':
        embedding_file_1 = f'{description}_{step}_astnn_embedding.pkl'
        embedding_file_2 = f'{description}_{step}_codebert_embedding.pkl'
    elif embedding_type == 'astnn+glove':
        embedding_file_1 = f'{description}_{step}_astnn_embedding.pkl'
        embedding_file_2 = f'{description}_{step}_glove_embedding.pkl'
    elif embedding_type == 'astnn+codebert+stereotype':
        embedding_file_1 = f'{description}_{step}_astnn_embedding.pkl'
        embedding_file_2 = f'{description}_{step}_codebert_embedding.pkl'
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'new_{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读 embedding
        embedding_list_1 = pd.read_pickle(join(model_path, embedding_file_1))
        embedding_list_2 = pd.read_pickle(join(model_path, embedding_file_2))
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        base = 0
        # seed_num_list = list()
        if len(graphs) == 0:
            continue
        # for graph in graphs:
        #     count = 0
        #     vertices = graph.find('vertices')
        #     vertex_list = vertices.findall('vertex')
        #     for vertex in vertex_list:
        #         if vertex.get('seed') == '1':
        #             count += 1
        #     seed_num_list.append(count)
        # one_seed_index = [i for i, x in enumerate(seed_num_list) if x == 1]
        # remain_seed_list = list()
        # for i in range(len(one_seed_index) - 1):
        #     remain_seed_list.append(math.floor((one_seed_index[i] + one_seed_index[i + 1]) / 2))
        # remain_seed_list.append(math.ceil((one_seed_index[-1] + seed_num_list[-1]) / 2))
        for graph in graphs:
            # 创建保存nodes和edges的数据结构
            nodes_tsv = list()
            edges_tsv = list()
            true_edge = 0
            true_node = 0
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')])
                _id = int(vertex.get('id'))
                # 去找 embedding 并添加  这儿不知道为什么会多一层列表
                # embedding_list_1[embedding_list_1['id'] == node_id]['embedding']
                # print(node_id)
                embedding_1 = embedding_list_1[embedding_list_1['id'] == node_id]['embedding'].iloc[0]
                embedding_2 = embedding_list_2[embedding_list_2['id'] == node_id]['embedding'].iloc[0]
                # embedding_seed = SEED_EMBEDDING(torch.tensor(int(vertex.get('seed')))).squeeze().tolist()
                # 进行组合
                embedding = embedding_1.tolist() + embedding_2.tolist()
                # embedding = embedding_1.tolist() + embedding_2.tolist() + embedding_seed
                if embedding_type == 'astnn+codebert':
                    embedding = embedding
                elif embedding_type == 'astnn+codebert+stereotype':
                    stereotype = vertex.get('stereotype')
                    embedding_stereotype = STEREOTYPE_EMBEDDING(
                        torch.tensor(step_1.index(stereotype))).squeeze().tolist()
                    embedding = embedding + embedding_stereotype
                # nodes_tsv.append(
                #     [_id, embedding, int(vertex.get('origin')), vertex.get('kind'), int(vertex.get('seed'))])
                nodes_tsv.append(
                    [_id, embedding, int(vertex.get('origin')), vertex.get('kind'), int(vertex.get('origin'))])
                if int(vertex.get('origin')) == 1:
                    true_node += 1
            edges = graph.find('edges')
            edge_list = edges.findall('edge')
            for edge in edge_list:
                start, end, edge_label = adjust_edge(edge, vertex_list)
                edges_tsv.append([start, end, edge_label])
                # edges_tsv.append([end, start, edge_label + 6]) # 加入反向边
                if int(edge.get('origin')) == 1:
                    true_edge += 1
            dest = join(dest_path, f'model_dataset_{str(step)}', dataset,
                        f'{project_model_name}_{model_dir}_{str(base)}')
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(dest)
            # if graphs.index(graph) in remain_seed_list and len(nodes_tsv) > 0 and len(edges_tsv) > 0:
            if len(nodes_tsv) > 0 and len(edges_tsv) > 0:
                if true_node > 1:
                    pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']).to_csv(
                        join(dest, 'nodes.tsv'), index=False)
                    pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation']).to_csv(
                        join(dest, 'edges.tsv'), index=False)
            base += 1


def save_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name, embedding_type, description):
    if embedding_type == 'astnn':
        embedding_file = f'{description}_{step}_astnn_embedding.pkl'
    elif embedding_type == 'glove':
        embedding_file = f'{description}_{step}_glove_embedding.pkl'
    elif embedding_type == 'codebert':
        embedding_file = f'{description}_{step}_codebert_embedding.pkl'
    elif embedding_type == 'astnn+codebert' or embedding_type == 'astnn+glove' or embedding_type == 'astnn+codebert+stereotype':
        save_composed_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name, embedding_type,
                            description)
        return
    else:
        embedding_file = f'{description}_{step}_astnn_embedding.pkl'
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'new_{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读 embedding
        embedding_list = pd.read_pickle(join(model_path, embedding_file))
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        base = 0
        # seed_num_list = list()
        if len(graphs) == 0:
            continue
        # for graph in graphs:
        #     count = 0
        #     vertices = graph.find('vertices')
        #     vertex_list = vertices.findall('vertex')
        #     for vertex in vertex_list:
        #         if vertex.get('seed') == '1':
        #             count += 1
        #     seed_num_list.append(count)
        # one_seed_index = [i for i, x in enumerate(seed_num_list) if x == 1]
        # remain_seed_list = list()
        # for i in range(len(one_seed_index) - 1):
        #     remain_seed_list.append(math.floor((one_seed_index[i] + one_seed_index[i + 1]) / 2))
        # remain_seed_list.append(math.ceil((one_seed_index[-1] + seed_num_list[-1]) / 2))
        for graph in graphs:
            # 创建保存nodes和edges的数据结构
            nodes_tsv = list()
            edges_tsv = list()
            true_edge = 0
            true_node = 0
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            # print(embedding_list)
            for vertex in vertex_list:
                node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')])
                _id = int(vertex.get('id'))
                # print(node_id)
                # 去找 embedding 并添加  这儿不知道为什么会多一层列表
                embedding = embedding_list[embedding_list['id'] == node_id]['embedding'].iloc[0].tolist()
                nodes_tsv.append(
                    [_id, embedding, int(vertex.get('origin')), vertex.get('kind'), int(vertex.get('origin'))])
                # embedding = embedding + SEED_EMBEDDING(torch.tensor(int(vertex.get('seed')))).squeeze().tolist()
                # nodes_tsv.append(
                #     [_id, embedding, int(vertex.get('origin')), vertex.get('kind'), int(vertex.get('seed'))])
                if int(vertex.get('origin')) == 1:
                    true_node += 1
            edges = graph.find('edges')
            edge_list = edges.findall('edge')
            for edge in edge_list:
                start, end, edge_label = adjust_edge(edge, vertex_list)
                edges_tsv.append([start, end, edge_label])
                # edges_tsv.append([end, start, edge_label + 6]) # 加入反向边
                if int(edge.get('origin')) == 1:
                    true_edge += 1
            dest = join(dest_path, f'model_dataset_{str(step)}', dataset,
                        f'{project_model_name}_{model_dir}_{str(base)}')
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(dest)
            # 如果没有节点，或者没有边，或者节点数小于step 都需要过滤掉,也就是stimulation
            # if graphs.index(graph) in remain_seed_list and len(nodes_tsv) > 0 and len(edges_tsv) > 0:
            if len(nodes_tsv) > 0 and len(edges_tsv) > 0:
                if true_node > 1:
                    pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']).to_csv(
                        join(dest, 'nodes.tsv'), index=False)
                    pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation']).to_csv(
                        join(dest, 'edges.tsv'), index=False)
                # if dataset != 'train':
                #     if true_node > step:
                #         pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']).to_csv(
                #             join(dest, 'nodes.tsv'), index=False)
                #         pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation']).to_csv(
                #             join(dest, 'edges.tsv'), index=False)
                # else:
                #     pd.DataFrame(nodes_tsv, columns=['node_id', 'code_embedding', 'label', 'kind', 'seed']).to_csv(
                #         join(dest, 'nodes.tsv'), index=False)
                #     pd.DataFrame(edges_tsv, columns=['start_node_id', 'end_node_id', 'relation']).to_csv(
                #         join(dest, 'edges.tsv'), index=False)
            base += 1


def main_func(description: str, step: int, dest_path: str, embedding_type: str):
    """
    构建并拆分数据集
    all: 四个项目分别拆分成train,valid,test : 比例8:1:1 \n
    onlymylyn: mylyn作为train, 其余三个项目各自拆分为valid,test=1:1
    nopde:pde拆分为valid:test=1:1,其余为train
    noplatform: platform差分为valid:test=1:1,其余为train

    :param description: all, onlymylyn, nopde, noplatform
    :param step: 步长
    :param dest_path: 数据集保存的路径
    :return:
    """
    if os.path.exists(join(dest_path, f'model_dataset_{str(step)}')):
        shutil.rmtree(join(dest_path, f'model_dataset_{str(step)}'))
    if description == 'all':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        for project_model_name in project_model_list:
            project_path = join(root_path, project_model_name, 'repo_first_3')
            for dataset in dataset_ratio:
                ratios = dataset_ratio.get(dataset)
                model_dir_list = get_models_by_ratio(project_model_name, ratios[0], ratios[1])
                model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
                save_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name, embedding_type,
                           description)
    elif description == 'onlymylyn':
        project_model_name = 'my_mylyn'
        project_path = join(root_path, project_model_name, 'repo_first_3')
        model_dir_list = get_models_by_ratio(project_model_name, 0, 1)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        save_model(project_path, model_dir_list, step, dest_path, 'train', project_model_name, embedding_type,
                   description)
        for project_model_name in ['my_pde', 'my_platform']:
            project_path = join(root_path, project_model_name, 'repo_first_3')
            model_dir_list = get_models_by_ratio(project_model_name, 0, 0.5)
            model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
            save_model(project_path, model_dir_list, step, dest_path, 'valid', project_model_name, embedding_type,
                       description)
            model_dir_list = get_models_by_ratio(project_model_name, 0.5, 1)
            model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
            save_model(project_path, model_dir_list, step, dest_path, 'test', project_model_name, embedding_type,
                       description)
    elif description == 'nopde':
        project_model_name = 'my_pde'
        project_path = join(root_path, project_model_name, 'repo_first_3')
        model_dir_list = get_models_by_ratio(project_model_name, 0, 0.5)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        save_model(project_path, model_dir_list, step, dest_path, 'valid', project_model_name, embedding_type,
                   description)
        model_dir_list = get_models_by_ratio(project_model_name, 0.5, 1)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        save_model(project_path, model_dir_list, step, dest_path, 'test', project_model_name, embedding_type,
                   description)
        for project_model_name in ['my_mylyn', 'my_platform']:
            project_path = join(root_path, project_model_name, 'repo_first_3')
            model_dir_list = get_models_by_ratio(project_model_name, 0, 1)
            model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
            save_model(project_path, model_dir_list, step, dest_path, 'train', project_model_name, embedding_type,
                       description)
    elif description == 'noplatform':
        project_model_name = 'my_platform'
        project_path = join(root_path, project_model_name, 'repo_first_3')
        model_dir_list = get_models_by_ratio(project_model_name, 0, 0.5)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        save_model(project_path, model_dir_list, step, dest_path, 'valid', project_model_name, embedding_type,
                   description)
        model_dir_list = get_models_by_ratio(project_model_name, 0.5, 1)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        save_model(project_path, model_dir_list, step, dest_path, 'test', project_model_name, embedding_type,
                   description)
        for project_model_name in ['my_mylyn', 'my_pde']:
            project_path = join(root_path, project_model_name, 'repo_first_3')
            model_dir_list = get_models_by_ratio(project_model_name, 0, 1)
            model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
            save_model(project_path, model_dir_list, step, dest_path, 'train', project_model_name, embedding_type,
                       description)
    elif description == 'mylyn':
        project_model_list = ['my_mylyn']
        for project_model_name in project_model_list:
            project_path = join(root_path, project_model_name, 'repo_first_3')
            for dataset in mylyn_dataset_ratio:
                ratios = mylyn_dataset_ratio.get(dataset)
                model_dir_list = get_models_by_ratio(project_model_name, ratios[0], ratios[1])
                model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
                save_model(project_path, model_dir_list, step, dest_path, dataset, project_model_name, embedding_type,
                           description)
