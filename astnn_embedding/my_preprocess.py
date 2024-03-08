import os
from os.path import join

import pandas as pd
from javalang.tree import FieldDeclaration

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code',
                 'my_mylyn', 'repo_first_3', '1013')

# java_tsv = pd.read_csv(join(root_path, 'processed_java_codes.tsv'), delimiter='\t')
# print(java_tsv)

ast_df = pd.read_pickle(join(root_path, 'astnn_ast.pkl'))
dec: FieldDeclaration = ast_df.iloc[3]['code']
print(dec.children)

# train_vector_df = pd.read_pickle(join(root_path, 'astnn_embedding.pkl'))
# print(train_vector_df)
