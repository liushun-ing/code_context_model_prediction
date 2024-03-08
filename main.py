"""
run python file
import the file you want to run, and then run main.py.
"""
import dgl
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    print('main...')
    #
    #
    # g = dgl.graph(((0, 2, 3, 5, 2, 0),
    #                (2, 2, 5, 2, 2, 4)))
    # output = torch.tensor([0.4, 0.5, 1, 0.2, 0.3, 0.5])
    # indices = (output >= 0.4).nonzero(as_tuple=True)[0].tolist()
    # # sub_g = dgl.node_subgraph(g, indices.tolist())
    # print(g, output, indices)
    # tag = np.zeros(len(indices))
    # for i in range(len(indices)):
    #     if tag[i] != 1:
    #         for j in range(len(indices)):
    #             if i != j:
    #                 if g.has_edges_between(indices[i], indices[j]):
    #                     tag[i], tag[j] = 1, 1
    # print(tag)
    # for t in range(len(tag)):
    #     if tag[t] == 0:
    #         output[t] = 0.0
    # print(output)

    # 创建一个示例的 DataFrame
    # data = {'ID': [1, 2, 3, 4, 5],
    #         'Value': ['A', 'B', 'C', 'D', 'E']}
    # df = pd.DataFrame(data)
    # # 设置要进行的多次采样次数
    # num_samples = 2
    # # 进行多次不重复采样
    # for _ in range(num_samples):
    #     sampled_df = df.sample(n=2, replace=False).copy(deep=True)
    #     # 使用 merge 方法找到两个 DataFrame 中相同的行
    #     merged_df = pd.merge(df, sampled_df, how='outer', indicator=True)
    #     merged_df = merged_df.loc[lambda x: x['_merge'] == 'left_only']
    #     df = merged_df.drop(columns=['_merge'])
    #     # 打印每次的采样结果
    #     print("采样结果：")
    #     print(sampled_df)
    #     print("---")





