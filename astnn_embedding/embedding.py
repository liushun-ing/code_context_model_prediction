"""
根据 word2vec 模型和 block sequence 生成词向量
"""
import copy
import os
from os.path import join

import pandas as pd
import numpy as np
import torch
from javalang.tree import FieldDeclaration, ConstructorDeclaration, ClassDeclaration, InterfaceDeclaration, \
    MethodDeclaration, ConstantDeclaration
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import warnings
from gensim.models.word2vec import Word2Vec

from my_model import BatchProgramCC

tqdm.pandas()
warnings.filterwarnings('ignore')

repo_root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation',
                      'git_repo_code')
root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'astnn_embedding')


def choose_prediction_step(step: int, ast: pd.DataFrame, model_path, model_dir):
    """
    根据 step 过滤 ast

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


def generate_block_seqs(code_ast, vocab, max_token) -> list[list]:
    """
    generate block sequences with index representations

    Returns:
    """
    from utils import get_blocks_v1

    def tree_to_index(node):
        token = node.token
        # 如果在词汇表中，就用词汇表的token，否则为最大的token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        """transfer tree to word token sequences

        :arg r: ast tree
        """
        # print('ast tree: ', r)
        blocks = []
        get_blocks_v1(r, blocks)  # 这里应该是根据ast tree去得到子树的一系列表达
        # print('get_blocks', r, '--------', blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)  # 然后去转换为vocab token形式
            tree.append(btree)
        # print(tree)
        return tree

    blocks = trans2seq(code_ast)  # 将code转换为sequence
    return blocks  # 这里实际上是用于去预测的处理过的数据，是code word2vec sequence


def pooling_class_interface_embedding(tree, model, vocab, max_token):
    # print(tree)
    embeddings = []
    chi = copy.deepcopy(tree.children[4])
    tree.children[4].clear()
    # embedding first using self information, prevent the error from no body
    embeddings.append(model.encode(generate_block_seqs(tree, vocab, max_token)))
    for i in chi:
        tree.children[4].append(i)
    children = tree.children[4]
    for c in children:
        if isinstance(c, FieldDeclaration) or isinstance(c, ConstantDeclaration):
            embeddings.append(model.encode(generate_block_seqs(c, vocab, max_token)))
        elif isinstance(c, MethodDeclaration) or isinstance(c, ConstructorDeclaration):
            embeddings.append(model.encode(generate_block_seqs(c, vocab, max_token)))
        elif isinstance(c, ClassDeclaration) or isinstance(c, InterfaceDeclaration):
            embeddings.append(pooling_class_interface_embedding(c, model, vocab, max_token))
    # print(len(embeddings))
    # print(torch.cat(embeddings))
    return torch.mean(torch.stack(embeddings), dim=0)
    # return torch.max(torch.stack(embeddings), dim=0).values


def get_embedding(ast_df: pd.DataFrame, model, vocab, max_token) -> pd.DataFrame:
    """
    读取 block 文件，获取每个 element's 200 embedding vector，并保存到 vector.pkl 文件中
    """
    print(f'embedding code...->{ast_df.shape}')

    def embedding(member):
        """
        需要针对不同的类型进行不同的embedding code
        function and variable directly embedding，class and interface need to pooling

        :param member: [id, ast]
        :return: code embedding
        """
        _id: str = member[0]
        _id = _id[_id.find('_') + 1:]
        code_type = _id[:_id.find('_')]
        code_ast = member[1]
        # print(member)
        if code_type == 'class' or code_type == 'interface':
            # first get embedding of all fields and methods, and then max pooling
            final_embedding = pooling_class_interface_embedding(code_ast, model, vocab, max_token)
        else:
            final_embedding = model.encode(generate_block_seqs(code_ast, vocab, max_token))
        return final_embedding

    for i, row in ast_df.iterrows():
        ast_df.at[i, 'code'] = embedding(row)
    ast_df.columns = ['id', 'embedding']
    return ast_df


def main_func(step, description, r=0.8, use_gpu=True, hidden_dim=100, code_dim=128):
    if description == 'all':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        ratio = r
    elif description == 'onlymylyn':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        ratio = 1
    elif description == 'nopde':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        ratio = 1
    elif description == 'noplatform':
        project_model_list = ['my_pde', 'my_platform', 'my_mylyn']
        ratio = 1
    elif description == 'mylyn':
        project_model_list = ['my_mylyn']
        ratio = r
    else:
        project_model_list = []
        ratio = 1
    word2vec = Word2Vec.load(join(root_path, 'w2v', f'{description}_{str(step)}_{str(ratio)}_node_w2v_{code_dim}')).wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    # 设置词汇表
    vocab = word2vec.vocab
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    print('embeddings', embeddings, embeddings.shape)
    print('vocab', len(vocab))
    HIDDEN_DIM = hidden_dim  # final embedding size = HIDDEN_DIM * 2
    ENCODE_DIM = code_dim
    BATCH_SIZE = 1
    USE_GPU = use_gpu
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, BATCH_SIZE,
                           USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()
        # model.to('cuda:1')
    model.hidden = model.init_hidden()
    for project_model_name in project_model_list:
        print('**********', project_model_name)
        project_path = join(repo_root_path, project_model_name, 'repo_first_3')
        model_dir_list = os.listdir(project_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        for model_dir in model_dir_list:
            print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            ast_path = join(model_path, 'astnn_ast.pkl')
            # 如果不存在ast，跳过处理
            if not os.path.exists(ast_path):
                continue
            ast_df = pd.read_pickle(ast_path)
            ast_df = choose_prediction_step(step, ast_df, model_path, model_dir)
            get_embedding(ast_df, model, vocab, MAX_TOKENS).to_pickle(join(model_path, f'{description}_{step}_astnn_embedding.pkl'))
