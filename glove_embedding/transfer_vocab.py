import os
from os.path import join

from gensim.test.utils import datapath, get_tmpfile

glove_root = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'glove_embedding')


def transfer_vocab(step, description, r):
    # 输入文件
    glove_file = datapath(join(glove_root, 'trained_model', f'{description}_{step}_{r}', 'vectors.txt'))
    # 输出文件
    tmp_file = get_tmpfile(join(glove_root, 'trained_model', f'{description}_{step}_{r}', 'w2v_vectors.txt'))
    # call glove2word2vec script
    # default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)


def main_func(step: int, description, r=0.8):
    transfer_vocab(step, description, r)
