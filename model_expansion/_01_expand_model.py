"""
该程序用来针对code context model进行扩展，得到用于训练的大图集合
分1-step,2-step,3-step分别扩展，然后分别保存到相应的文件中
"""
import os
from os.path import join

import model_loader
from xmlparser.doxygen_main import get_relations, expand_graph

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def main_func(project_model_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # index = model_dir_list.index('1190')
    for model_dir in model_dir_list[model_dir_list.index('4885'):]:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # if os.path.exists(join(model_path, '1_step_expanded_model.xml')):
        #     continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        # 分别生成1_step_expanded_model.xml,2_step_expanded_model.xml,3_step_expanded_model.xml
        # 加载doxygen矩阵
        all_repo_metrics = get_relations.solve_doxygen_metrics(model_path)
        print('----metrics loaded')
        # 1 step扩展
        # 读code context model
        model_graphs = model_loader.load_code_context_model(model_file)
        expand_graph.expand_model(model_graphs, all_repo_metrics, 1)
        model_loader.save_expanded_model(model_graphs, join(model_path, '1_step_expanded_model.xml'))
        # # 2 step扩展 需要重置
        # model_graphs = model_loader.load_code_context_model(model_file)
        # expand_graph.expand_model(model_graphs, all_repo_metrics, 2)
        # model_loader.save_expanded_model(model_graphs, join(model_path, '2_step_expanded_model.xml'))
        # # 3 step扩展
        # model_graphs = model_loader.load_code_context_model(model_file)
        # expand_graph.expand_model(model_graphs, all_repo_metrics, 3)
        # model_loader.save_expanded_model(model_graphs, join(model_path, '3_step_expanded_model.xml'))


# ecf
# main_func('my_ecf')
# pde
# main_func('my_pde')
# platform
# main_func('my_platform')
# # mylyn
# print(root_path)
main_func('my_mylyn')
