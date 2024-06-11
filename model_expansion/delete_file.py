import os

from os.path import join

import warnings
from tqdm.auto import tqdm

tqdm.pandas()
warnings.filterwarnings('ignore')

repo_root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation',
                      'git_repo_code')


def main_func():
    for project_model_name in ['my_pde', 'my_platform', 'my_mylyn']:
        print('************', project_model_name)
        project_path = join(repo_root_path, project_model_name, 'repo_first_3')
        model_dir_list = os.listdir(project_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        for model_dir in model_dir_list:
            print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            file_path = join(model_path, 'mylyn_3_codebert_embedding.pkl')
            # 如果不存在ast，跳过处理
            if os.path.exists(file_path):
                print('---------------', file_path)
                # os.remove(file_path)
                os.rename(file_path, join(model_path, '3_codebert_embedding.pkl'))


if __name__ == '__main__':
    main_func()
