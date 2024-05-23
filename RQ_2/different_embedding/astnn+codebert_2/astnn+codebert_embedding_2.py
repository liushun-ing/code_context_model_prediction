import os
from os.path import join

from pathlib import Path
import argparse

os.sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))

from GNN_Node_Classification import construct_input, node_classification, test_model
import nni

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--nni", type=bool, required=False, default=False)
parser.add_argument("--step", type=int, required=False, default=2)
parser.add_argument("--gpu", type=str, required=False, default='4')
args = parser.parse_args()

construct = True
load_lazy = True

my_params = {
    'description': 'mylyn',
    'embedding_type': 'astnn+codebert',
    'current_path': join(os.path.dirname(os.path.realpath(__file__))),  # save to current dir
    'step': args.step,
    'under_sampling_threshold': 15.0,
    'model_type': 'GCN',
    'num_layers': 3,
    'in_feats': 1280,
    'hidden_size': 512,
    'dropout': 0.1,
    'attention_heads': 8,
    'num_heads': 8,
    'num_edge_types': 6,
    'epochs': 80,
    'lr': 0.001,
    'batch_size': 16,
    'threshold': 0.4,
    'weight_decay': 1e-6,
    'approach': 'concat'
}

if not args.nni:
    assert args.gpu != ''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    optimized_params = nni.get_next_parameter()
    params = my_params.keys()
    for param in params:
        if param in optimized_params:
            my_params[param] = optimized_params[param]

my_params[
    "result_name"] = f"{my_params['description']}_{my_params['model_type']}_{my_params['embedding_type']}_{my_params['under_sampling_threshold']}_model"

print(my_params)
# construct input: train, valid, test dataset of four project
if construct:
    construct_input.main_func(
        description=my_params['description'],
        step=my_params['step'],
        dest_path=my_params['current_path'],
        embedding_type=my_params['embedding_type'],
        under_sampling_threshold=my_params['under_sampling_threshold']
    )

# train and save model
node_classification.main_func(
    save_path=my_params['current_path'],
    save_name=my_params['result_name'],
    step=my_params['step'],
    under_sampling_threshold=my_params['under_sampling_threshold'],
    model_type=my_params['model_type'],
    num_layers=my_params['num_layers'],
    in_feats=my_params['in_feats'],
    hidden_size=my_params['hidden_size'],
    dropout=my_params['dropout'],
    attention_heads=my_params['attention_heads'],
    num_heads=my_params['num_heads'],
    num_edge_types=my_params['num_edge_types'],
    epochs=my_params['epochs'],
    lr=my_params['lr'],
    batch_size=my_params['batch_size'],
    threshold=my_params['threshold'],
    weight_decay=my_params['weight_decay'],
    approach=my_params['approach'],
    load_lazy=load_lazy,
    use_nni=args.nni
)

# show train result
# view.main_func(
#     result_path=current_path,
#     step=step,
#     load_name=result_name
# )

# test model
test_model.main_func(
    model_path=my_params['current_path'],
    load_name=my_params['result_name'],
    step=my_params['step'],
    model_type=my_params['model_type'],
    num_layers=my_params['num_layers'],
    in_feats=my_params['in_feats'],
    hidden_size=my_params['hidden_size'],
    attention_heads=my_params['attention_heads'],
    num_heads=my_params['num_heads'],
    num_edge_types=my_params['num_edge_types'],
    load_lazy=load_lazy,
    approach=my_params['approach'],
    use_nni=args.nni
)
