import argparse
import os
from os.path import join

from GNN_Node_Classification import data_estimator, construct_input, node_classification, view, test_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU编号

# see the sample ratios of four project
# data_estimator.estimate_positive_and_negative_samples(['my_pde', 'my_platform', 'my_mylyn'], step=3)

# parameters
description = 'mylyn'
embedding_type = 'astnn+codebert'
current_path = join(os.path.dirname(os.path.realpath(__file__)))  # save to current dir

steps = [1]
construct = False
load_lazy = False

args = {
    'under_sampling_threshold': 15.0,
    'model_type': 'GCN',
    'num_layers': 3,
    'in_feats': 1280,
    'hidden_size': 1024,
    'dropout': 0.1,
    'attention_heads': 10,
    'num_heads': 8,
    'num_edge_types': 6,
    'epochs': 80,
    'lr': 0.001,
    'batch_size': 16,
    'threshold': 0.4,
    'weight_decay': 1e-6,
    'approach': 'attention'
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
    result_name = f'{description}_{args["model_type"]}_{embedding_type}_{args["under_sampling_threshold"]}_model'

    for step in steps:
        print(f'step: {step}', args)
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
            under_sampling_threshold=args['under_sampling_threshold'],
            model_type=args['model_type'],
            num_layers=args['num_layers'],
            in_feats=args['in_feats'],
            hidden_size=args['hidden_size'],
            dropout=args['dropout'],
            attention_heads=args['attention_heads'],
            num_heads=args['num_heads'],
            num_edge_types=args['num_edge_types'],
            epochs=args['epochs'],
            lr=args['lr'],
            batch_size=args['batch_size'],
            threshold=args['threshold'],
            weight_decay=args['weight_decay'],
            approach=args['approach'],
            load_lazy=load_lazy,
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
            model_type=args['model_type'],
            num_layers=args['num_layers'],
            in_feats=args['in_feats'],
            hidden_size=args['hidden_size'],
            attention_heads=args['attention_heads'],
            num_heads=args['num_heads'],
            num_edge_types=args['num_edge_types'],
            load_lazy=load_lazy,
            approach=args['approach']
        )


if __name__ == '__main__':
    run()
