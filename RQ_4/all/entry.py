import os
from os.path import join

from GNN_Node_Classification import data_estimator, construct_input, node_classification, view, test_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU编号

# see the sample ratios of four project
# data_estimator.estimate_positive_and_negative_samples(['my_pde', 'my_platform', 'my_mylyn'], step=2)

# parameters
description = 'all'
embedding_type = 'astnn'
current_path = join(os.path.dirname(os.path.realpath(__file__)))  # save to current dir
# model_name = ['GCN3'][0]
under_sampling_threshold = 40.0
code_embedding = 200
epochs = 80
lr = 0.001
batch_size = 16
hidden_size = 64
out_feats = 16
dropout = 0.2
threshold = 0.4
model_name = 'GCN3'

for step in [2]:
    # construct input: train, valid, test dataset of four project
    # construct_input.main_func(
    #     description=description,
    #     step=step,
    #     dest_path=current_path,
    #     embedding_type=embedding_type
    # )
    # for model_name in ['GCN2', 'GCN3', 'GCN4', 'RGCN2', 'RGCN3', 'RGCN4', 'GAT2', 'GAT3', 'GAT4', 'GraphSAGE2', 'GraphSAGE3', 'GraphSAGE4']:
    # for model_name in ['GCN2', 'GCN3', 'GCN4']:
    # for under_sampling_threshold in [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0]:
    result_name = f'{description}_{model_name}_{embedding_type}_{under_sampling_threshold}_model'
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
        out_feats=out_feats,
    )

