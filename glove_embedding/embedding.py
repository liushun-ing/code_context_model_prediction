"""
根据 glove 模型生成词向量
"""
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm

import warnings

tqdm.pandas()
warnings.filterwarnings('ignore')

repo_root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation',
                      'git_repo_code')
glove_root = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'glove_embedding')


def choose_prediction_step(step: int, sources: pd.DataFrame, model_path, model_dir):
    """
    根据 step 过滤 ast

    :param sources: 完整的 ast 数据
    :param model_path: model 所在路径
    :param model_dir: model 序号
    :return: 过滤后的 ast
    """
    if step == 3:
        return sources
    else:
        step_model_path = join(model_path, str(step) + '_step_expanded_model.xml')
        tree = ET.parse(step_model_path)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        keys = []
        for graph in graphs:
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                ref_id = vertex.get('ref_id')
                kind = vertex.get('kind')
                keys.append(model_dir + '_' + kind + '_' + ref_id)
        return sources[sources['id'].isin(keys)]


def get_all(glove_model, tokens):
    # 分词
    words = tokens.split()
    res = []
    # 计算段落的总词向量
    for word in words:
        if word in glove_model:
            res.append(glove_model[word])
        else:  # if OOV random generate a tensor
            res.append(torch.randn(512).tolist())
    return torch.tensor(res)


def get_embedding(tokens: pd.DataFrame, model) -> pd.DataFrame:
    print('embedding glove code...')
    tokens['tokens'] = tokens['tokens'].progress_apply(lambda x: torch.mean(get_all(model, x), dim=0))
    tokens.columns = ['id', 'embedding']
    return tokens


def main_func(step, description, r=0.8):
    if description == 'all':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        ratio = r
    elif description == 'onlymylyn':
        project_model_list = ['my_mylyn']
        ratio = 0.8
    elif description == 'nopde':
        project_model_list = ['my_platform', 'my_mylyn']
        ratio = 0.8
    elif description == 'noplatform':
        project_model_list = ['my_pde', 'my_mylyn']
        ratio = 0.8
    elif description == 'mylyn':
        project_model_list = ['my_mylyn']
        ratio = r
    else:
        project_model_list = []
        ratio = 1
    # 加载转化后的文件
    model_file = join(glove_root, 'trained_model', f'{description}_{step}_{ratio}', 'w2v_vectors.txt')
    model = KeyedVectors.load_word2vec_format(model_file)
    for project_model_name in project_model_list:
        print('**********', project_model_name)
        project_path = join(repo_root_path, project_model_name, 'repo_first_3')
        model_dir_list = os.listdir(project_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        for model_dir in model_dir_list:
            print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            tokens_path = join(model_path, 'java_tokens.tsv')
            # 如果不存在block_path，跳过处理
            if not os.path.exists(tokens_path):
                continue
            sources = pd.read_csv(tokens_path, sep='\t')
            sources = choose_prediction_step(step, sources, model_path, model_dir)
            get_embedding(sources, model).to_pickle(
                join(model_path, f'{description}_{step}_glove_embedding.pkl'))
