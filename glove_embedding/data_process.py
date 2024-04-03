"""
获取所有的vocab句子
"""
import os
from os.path import join
import xml.etree.ElementTree as ET

import pandas as pd
import warnings
from tqdm.auto import tqdm

from dataset_split_util import get_models_by_ratio

tqdm.pandas()
warnings.filterwarnings('ignore')

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')
glove_root = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'glove_embedding')


def choose_prediction_step(step: int, tokens: pd.DataFrame, model_path, model_dir):
    """
    根据 step 过滤 ast

    :param step: 步长
    :param tokens: 完整的 ast 数据
    :param model_path: model 所在路径
    :param model_dir: model 序号
    :return: 过滤后的 ast
    """
    if step == 3:
        return tokens
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
        return tokens[tokens['id'].isin(keys)]


def save_train_java_tokens(trees, step, ratio, description):
    """
    将 vocab 的句子写入到 txt 文件中

    Args:
        trees: 需要训练的 ast 集合 pd.DataFrame(['id', 'code'])
        step: 步长
        ratio: 训练的数据比例
        description: 训练的项目描述信息：all训练所有项目，onlymylyn，nopde,noplatform

    Returns: 无
    """
    if not os.path.exists(join(glove_root, 'train_vocab')):
        os.mkdir(join(glove_root, 'train_vocab'))

    txt_path = join(glove_root, 'train_vocab', f'{description}_{str(step)}_{str(ratio)}_vocab.txt')
    with open(txt_path, 'w') as vocab_file:
        trees['tokens'].progress_apply(lambda x: vocab_file.write(x + "\n"))
        vocab_file.close()


def main_func(step: int, description, r=0.8):
    if description == 'all':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        ratio = r
    elif description == 'onlymylyn':
        project_model_list = ['my_mylyn']
        ratio = 1
    elif description == 'nopde':
        project_model_list = ['my_platform', 'my_mylyn']
        ratio = 1
    elif description == 'noplatform':
        project_model_list = ['my_pde', 'my_mylyn']
        ratio = 1
    elif description == 'mylyn':
        project_model_list = ['my_mylyn']
        ratio = r
    else:
        project_model_list = []
        ratio = 1
    java_tokens = pd.DataFrame(columns=['id', 'tokens'])
    for project_model_name in project_model_list:
        print('***********************', project_model_name)
        project_path = join(root_path, project_model_name, 'repo_first_3')
        model_dir_list = get_models_by_ratio(project_model_name, 0.0, ratio)
        for model_dir in model_dir_list:
            print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            tokens_path = join(model_path, 'java_tokens.tsv')
            # 如果不存在ast，跳过处理
            if not os.path.exists(tokens_path):
                continue
            tokens = pd.read_csv(tokens_path, sep='\t')
            tokens = choose_prediction_step(step, tokens, model_path, model_dir)
            print(f'tokens size: {len(tokens)}')
            java_tokens = pd.concat([java_tokens, tokens])
    save_train_java_tokens(java_tokens, step, ratio, description)

