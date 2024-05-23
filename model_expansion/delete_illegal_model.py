"""
delete illegal model
将不支持预测的model删除，也就是连通分量中节点个数最大都是1的模型
"""
import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    count = 0
    for model_dir in model_dir_list:
        # print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        # model_file = join(model_path, '_code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            # 也删除
            shutil.rmtree(model_path)
            print(f'delete {model_file}')
            count += 1
        # 下面对code context model进行component拆分
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        graphs = root.findall("graph")
        if len(graphs) == 0:
            # delete
            shutil.rmtree(model_path)
            print(f'delete {model_file}')
            count += 1
    print(count)


if __name__ == '__main__':
    # ecf
    # main_func('my_ecf')
    # pde
    main_func('my_pde')
    # platform
    main_func('my_platform')
    # # mylyn
    main_func('my_mylyn')
