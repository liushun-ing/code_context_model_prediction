import shutil

import os
import xml.etree.ElementTree as ET
from os.path import join

from xmlparser.doxygen_main.get_standard_elements import solve_one


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    return directory


root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')
time_root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'repo_vs_commit_order', 'IQR_code_timestamp', '05')


def get_all_timestamp(time_path):
    tree = ET.parse(time_path)  # 拿到xml树
    root = tree.getroot()
    code_element = root.find('code_elements')
    elements = code_element.findall('element')
    time_list = dict()
    for element in elements:
        res = solve_one(element.text)
        time_list[res[1]] = element.get('event_start_date')
    return time_list


def main_func(project_model_name: str, project_name: str):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    time_file_path = join(time_root_path, project_name)
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_root = join(model_path, 'code_context_model.xml')
        if not os.path.exists(model_path):
            continue
        # 拿到所有时间戳
        time_model_path = join(time_file_path, f'{model_dir}.xml')
        all_timestamp = get_all_timestamp(time_model_path)
        tree = ET.parse(model_root)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        graphs = root.findall("graph")
        for graph in graphs:
            vertices = graph.find('vertices').findall('vertex')
            for vertex in vertices:
                if all_timestamp.get(vertex.get('label')) is not None:
                    vertex.set('timestamp', all_timestamp.get(vertex.get('label')))
                else:
                    print(vertex.get('label'))
                # print(all_timestamp.get(vertex.get('label')))
        tree.write(model_root)


main_func('my_mylyn', 'Mylyn')
main_func('my_platform', 'Platform')
main_func('my_pde', 'PDE')