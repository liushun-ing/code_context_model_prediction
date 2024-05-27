# """
# 由于类 declares 的字段和方法太多，对每个图中的 origin = 0 的字段和方法进行随机性选择
# 字段保留 20%， 方法保留 50%
# """
# import os
# import shutil
# from os.path import join
# import xml.etree.ElementTree as ET
#
# root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')
#
#
# def main_func(project_model_name: str, step):
#     project_path = join(root_path, project_model_name, 'repo_first_3')
#     model_dir_list = os.listdir(project_path)
#     # 读取code context model
#     model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
#     # index = model_dir_list.index('5232')
#     for model_dir in model_dir_list:
#         print('---------------', model_dir)
#         model_path = join(project_path, model_dir)
#         model_file = join(model_path, f'{step}_step_expanded_model.xml')
#         new_model_file = join(model_path, f'new_{step}_step_expanded_model.xml')
#         # 如果不存在模型，跳过处理
#         if not os.path.exists(model_file):
#             continue
#         # shutil.copy(rename_model_file, model_file)
#         # 下面对code context model进行component拆分
#         tree = ET.parse(model_file)  # 拿到xml树
#         # 获取XML文档的根元素
#         root = tree.getroot()
#         graphs = root.findall("graph")
#         for graph in graphs:
#             vertices = graph.find('vertices').findall('vertex')
#             edges = graph.find('edges').findall('edge')
#             for node in vertices:
#
#             for link in edges:
#
#         # 将XML写入文件
#         tree.write(new_model_file)
#
#
# if __name__ == '__main__':
#     # ecf
#     # main_func('my_ecf')
#     # pde
#     # main_func('my_pde')
#     # platform
#     # main_func('my_platform')
#     # # mylyn
#     # print(root_path)
#     main_func('my_mylyn', 1)
#     main_func('my_mylyn', 2)
#     main_func('my_mylyn', 3)
