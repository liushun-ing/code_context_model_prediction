import os
import subprocess
from os.path import join

from glove_embedding import data_process, transfer_vocab, embedding
glove_root = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'glove_embedding')


def train_glove(step, description, ratio):
    # 训练glove模型
    if not os.path.exists(join(glove_root, 'trained_model', f'{description}_{step}_{ratio}')):
        os.mkdir(join(glove_root, 'trained_model', f'{description}_{step}_{ratio}'))
    subprocess.run(f"cd glove && make && bash ./{description}_{step}_{ratio}.sh", shell=True)
    # subprocess.run("make", shell=True)
    # subprocess.run(f'bash {description}_{step}_{ratio}.sh', shell=True)


for step in [1]:
    description = 'mylyn'
    ratio = 0.8
    # 预处理数据
    print('process data')
    data_process.main_func(step, description, ratio)

    print('train glove model')
    train_glove(step, description, ratio)

    # 转换模型
    print('transfer model')
    transfer_vocab.main_func(step, description, ratio)

    # code embedding
    print('embedding glove code')
    embedding.main_func(step, description, ratio)
