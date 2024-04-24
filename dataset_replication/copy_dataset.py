"""
将code context model拆分component
"""
import os
import shutil
from os.path import join, isdir

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')
dest_path = '/data1/shunliu/dataset/'


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # index = model_dir_list.index('5232')
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        os.makedirs(join(dest_path, project_model_name, model_dir), exist_ok=True)
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        shutil.copy(model_file, join(dest_path, project_model_name, model_dir, 'code_context_model.xml'))
        for model_inner_file in os.listdir(model_path):
            model_inner_file_path = join(model_path, model_inner_file)
            if isdir(model_inner_file_path) and model_inner_file != "doxygen":
                shutil.copytree(model_inner_file_path, join(dest_path, project_model_name, model_dir, model_inner_file))



# ecf
# main_func('my_ecf')
# pde
main_func('my_pde')
# platform
main_func('my_platform')
# # mylyn
# print(root_path)
main_func('my_mylyn')
