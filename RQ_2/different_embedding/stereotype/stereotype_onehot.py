import os
from os.path import join

import stereotype_construct
from GNN_Node_Classification import data_estimator, node_classification, view, test_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU编号

# see the sample ratios of four project
# data_estimator.estimate_positive_and_negative_samples(['my_pde', 'my_platform', 'my_mylyn'], step=1)

# parameters
description = 'all'
embedding_type = 'stereotype'
current_path = join(os.path.dirname(os.path.realpath(__file__)))  # save to current dir
model_name = ['GCN3'][0]
under_sampling_threshold = [5.0, 10.0, 20.0, 40.0][3]  # 0表示不进行欠采样
code_embedding = 53
epochs = 50
lr = 0.001
result_name = f'{description}_{model_name}_{embedding_type}_{under_sampling_threshold}_model'
batch_size = 16
hidden_size = 64
out_feats = 16
dropout = 0.2
threshold = 0.4

for step in [1, 2, 3]:
    # construct input: train, valid, test dataset of four project
    # stereotype_construct.main_func(
    #     description=description,
    #     step=step,
    #     dest_path=current_path,
    # )

    # train and save model
    node_classification.main_func(
        save_path=current_path,
        save_name=result_name,
        step=step,
        under_sampling_threshold=under_sampling_threshold,
        model_name=model_name,
        code_embedding=code_embedding,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        hidden_size=hidden_size,
        out_feats=out_feats,
        dropout=dropout,
        threshold=threshold
    )

    # show train result
    # view.main_func(
    #     result_path=current_path,
    #     step=step,
    #     load_name=result_name
    # )

    # test model
    test_model.main_func(
        model_path=current_path,
        load_name=result_name,
        step=step,
        model_name=model_name,
        code_embedding=code_embedding,
        hidden_size=hidden_size,
        out_feats=out_feats
    )
