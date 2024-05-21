import argparse
import os
from os.path import join

from GNN_Node_Classification import data_estimator, construct_input, node_classification, view, test_model

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # GPU编号

# see the sample ratios of four project
# data_estimator.estimate_positive_and_negative_samples(['my_pde', 'my_platform', 'my_mylyn'], step=3)

# parameters
description = 'mylyn'
embedding_type = 'astnn+codebert'
current_path = join(os.path.dirname(os.path.realpath(__file__)))  # save to current dir
model_name = ['GCN3'][0]
threshold = 0.4
epochs = 80
steps = [1]
construct = False
load_lazy = True

args = {
    'under_sampling_threshold': 15.0,
    'code_embedding': 1280,
    'lr': 0.001,
    'batch_size': 16,
    'hidden_size': 1024,
    'hidden_size_2': 128,
    'out_feats': 64,
    'dropout': 0.1,
    'weight_decay': 1e-6
}


def run():
    # parser = argparse.ArgumentParser(description="Process super arguments.")
    # parser.add_argument('--under_sampling_threshold', type=int, default=15.0, help='欠采样比例')
    # parser.add_argument('--code_embedding', type=int, default=1280, help='初始维度')
    # parser.add_argument('--lr', type=int, default=0.0001, help='学习率')
    # parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    # parser.add_argument('--hidden_size', type=int, default=512, help='图注意力层 1')
    # parser.add_argument('--hidden_size_2', type=int, default=128, help='图注意力层 2')
    # parser.add_argument('--out_feats', type=int, default=64, help='图注意力层 3')
    # parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
    # parser.add_argument('--weight_decay', type=int, default=1e-6, help='weight_decay')
    # args = parser.parse_args()

    under_sampling_threshold = args['under_sampling_threshold']
    code_embedding = args['code_embedding']
    lr = args['lr']
    result_name = f'{description}_{model_name}_{embedding_type}_{under_sampling_threshold}_model'
    batch_size = args['batch_size']
    hidden_size = args['hidden_size']
    hidden_size_2 = args['hidden_size_2']
    out_feats = args['out_feats']
    dropout = args['dropout']
    weight_decay = args['weight_decay']

    for step in steps:
        print(f'model: {model_name}, step: {step}, epoch: {epochs}, undersampling: {under_sampling_threshold}, '
              f'hidden_size: {hidden_size}, hidden_size_2: {hidden_size_2}, out_feats: {out_feats}, lr: {lr}, '
              f'dropout: {dropout}, batch_size: {batch_size}, weight_decay: {weight_decay}')
        # construct input: train, valid, test dataset of four project
        if construct and not load_lazy:
            construct_input.main_func(
                description=description,
                step=step,
                dest_path=current_path,
                embedding_type=embedding_type
            )

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
            hidden_size_2=hidden_size_2,
            out_feats=out_feats,
            dropout=dropout,
            threshold=threshold,
            load_lazy=load_lazy,
            weight_decay=weight_decay
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
            hidden_size_2=hidden_size_2,
            out_feats=out_feats,
            load_lazy=load_lazy
        )


if __name__ == '__main__':
    run()
