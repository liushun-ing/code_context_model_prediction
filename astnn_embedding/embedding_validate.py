import os
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import mean

repo_root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')


# 1. transfer to tsv file
def transfer_files():
    s = ['97', '212', '1116']
    stereotypes = []
    for i in s:
        with open(f'./example/{i}_stereotype.txt') as f:
            lines = f.readlines()
            lines = [line.strip().split(' ') for line in lines]
            f.close()
        embeddings = pd.read_pickle(join(repo_root_path, 'my_mylyn', i, 'astnn_embedding.pkl'))
        for index in range(len(lines)):
            lines[index].append(torch.tensor(
                embeddings[embeddings['id'] == lines[index][0]]['embedding'].iloc[0].tolist()))
        stereotypes += lines
    new_df = pd.DataFrame(stereotypes, columns=['id', 'stereotype', 'astnn_embedding'])
    new_df.to_pickle(f'./example/stereotype.pkl')


def validate_embedding():
    df = pd.read_pickle(f'./example/stereotype.pkl')
    dot_result = []
    for index, (_id, _stereotype, _embedding) in df.iterrows():
        print(_id, _stereotype)
        new_df = df[df['id'] != _id]
        same_df = new_df[new_df['stereotype'] == _stereotype]
        not_same_df = new_df[new_df['stereotype'] != _stereotype]

        # print('df', same_df['stereotype'], not_same_df['stereotype'])
        def cosine_similar(x):
            # 余弦相似度
            # return F.cosine_similarity(x, _embedding, dim=0).tolist()
            # 改进余弦相似度
            # mean_t = torch.add(x, _embedding).div(2)
            # return F.cosine_similarity(x - mean_t, _embedding - mean_t, dim=0).tolist()
            # 欧氏距离
            # return F.pairwise_distance(x, _embedding).tolist()
            # pearson相关系数
            vx = x - torch.mean(x)
            vy = _embedding - torch.mean(_embedding)
            cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            return cost.tolist()
        same_result = same_df['embedding'].apply(cosine_similar)
        not_same_result = not_same_df['embedding'].apply(cosine_similar)
        print(
            f'same: {mean(same_result.values) if len(same_result.values) != 0 else 0}\n'
            f'not_same: {mean(not_same_result.values) if len(not_same_result.values) != 0 else 0}')
        dot_result.append((_id, _stereotype, same_result.values,
                           mean(same_result.values) if len(same_result.values) != 0 else 0, not_same_result.values,
                           mean(not_same_result.values) if len(not_same_result.values) != 0 else 0))
    pd.DataFrame(dot_result, columns=['id', 'stereotype', 'common', 'common_avg', 'uncommon', 'uncommon_avg']) \
        .to_csv('./example/result.tsv', index=False)


def view_result():
    df = pd.read_csv(f'./example/result.tsv')
    df = df[df['common_avg'] != 0]
    # 根据类别排序: class-function-variable
    df['id'] = df['id'].apply(lambda x: x[x.index('_')+1: x.find('_', x.index('_')+1)])
    df = df.sort_values(['id'])
    plt.plot(df['common_avg'].tolist(), '.--', label='common')
    plt.plot(df['uncommon_avg'].tolist(), '.--', label='uncommon')

    plt.legend()  # 显示上面的label
    plt.xlabel('id')  # x_label
    plt.ylabel('cosine similarity')
    plt.show()


if __name__ == '__main__':
    transfer_files()
    validate_embedding()
    view_result()
