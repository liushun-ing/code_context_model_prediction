"""
加载项目中的所有 ast,合并，并训练生成 word2vec 模型
可以选择步长，根据步长那部分数据训练word2vec模型
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
astnn_root = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'astnn_embedding')


def choose_prediction_step(step: int, ast: pd.DataFrame, model_path, model_dir):
    """
    根据 step 过滤 ast

    :param step: 步长
    :param ast: 完整的 ast 数据
    :param model_path: model 所在路径
    :param model_dir: model 序号
    :return: 过滤后的 ast
    """
    if step == 3:
        return ast
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
        return ast[ast['id'].isin(keys)]


def dictionary_and_embedding(trees, size, step, ratio, description):
    """
    分解词典和训练 word2vec 模型

    Args:
        trees: 需要训练的 ast 集合 pd.DataFrame(['id', 'code'])
        size: 嵌入大小 128
        step: 步长
        ratio: 训练的数据比例
        description: 训练的项目描述信息：all训练所有项目，onlymylyn，nopde,noplatform

    Returns: 无
    """
    if not os.path.exists(join(astnn_root, 'w2v')):
        os.mkdir(join(astnn_root, 'w2v'))
    from utils import get_sequence

    def trans_to_sequences(ast):
        sequence = []
        get_sequence(ast, sequence)
        return sequence

    corpus = trees['code'].progress_apply(trans_to_sequences)  # 调用函数，将 code 转换为 sequence
    # str_corpus = [' '.join(c) for c in corpus]  # 按照空格连接所有单词
    # 将这些 sentence 交给 word2vec 训练，并保存该模型
    from gensim.models.word2vec import Word2Vec
    print('training word2vec...')
    # w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
    w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=50000)
    print(len(w2v.wv.vocab))
    w2v.save(join(astnn_root, 'w2v', f'{description}_{str(step)}_{str(ratio)}_node_w2v_{str(size)}'))


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
    trees = pd.DataFrame(columns=['id', 'code'])
    for project_model_name in project_model_list:
        print('***********************', project_model_name)
        project_path = join(root_path, project_model_name, 'repo_first_3')
        model_dir_list = get_models_by_ratio(project_model_name, 0.0, ratio)
        for model_dir in model_dir_list:
            if model_dir == '2813':
                continue
            print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            ast_path = join(model_path, 'astnn_ast.pkl')
            # 如果不存在ast，跳过处理
            if not os.path.exists(ast_path):
                continue
            ast = pd.read_pickle(ast_path)
            ast = choose_prediction_step(step, ast, model_path, model_dir)
            print(f'ast size: {len(ast)}')
            trees = pd.concat([trees, ast])
    dictionary_and_embedding(trees, 128, step, ratio, description)

# Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
# corpus： 这是你的文本语料库。确保 corpus 是一个经过预处理的、可以迭代的文本集合，其中每个文档都是一个词列表。
# size： 这是 Word2Vec 模型的向量维度，即每个单词的嵌入向量的维度大小。
# workers： 这是用于训练的线程数。根据计算资源，你可以选择合适的值。在大多数情况下，选择一个较大的值（例如，16）是合适的，以便充分利用多核处理器。
# sg： 这是 Word2Vec 的训练算法选择，
#   sg=1 表示使用 Skip-Gram 算法，
#   sg=0 表示使用 CBOW（Continuous Bag of Words）算法。
#   通常，Skip-Gram 对于大型语料库和少量数据是更好的选择，而 CBOW 可以在小型数据集上更快地训练。
# max_final_vocab： 这是最终词汇表的最大大小，即最终保留的独特词汇数量。
