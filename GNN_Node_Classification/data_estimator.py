"""
用于做一些数据的评估工作
"""
import os
from os.path import join
import xml.etree.ElementTree as ET
import numpy as np

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def estimate_positive_and_negative_samples(project_model_list: list[str], step: int):
    total_res = []
    for project_model_name in project_model_list:
        print('!!!!!!!!!', project_model_name)
        project_path = join(root_path, project_model_name, 'repo_first_3')
        model_dir_list = os.listdir(project_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        ratios = []
        print(len(model_dir_list))
        for model_dir in model_dir_list:
            model_path = join(project_path, model_dir)
            model_file = join(model_path, f'new1_{step}_step_expanded_model.xml')
            # 如果不存在模型，跳过处理
            if not os.path.exists(model_file):
                continue
            tree = ET.parse(model_file)  # 拿到xml树
            # 获取XML文档的根元素
            code_context_model = tree.getroot()
            graphs = code_context_model.findall("graph")
            for graph in graphs:
                count = 0
                vertices = graph.find('vertices')
                vertex_list = vertices.findall('vertex')
                total = len(vertex_list)
                for vertex in vertex_list:
                    # print(vertex.get('origin'))
                    if vertex.get('origin') == '1':
                        count += 1
                if count > 1 and total > 0:
                    ratios.append((total - count) / (count if count > 0 else 1))
        total_res += ratios
        res = np.array(ratios)
        sort_counts_list = np.sort(res)
        Q3 = np.percentile(sort_counts_list, 75, method='midpoint')
        print(f'total: {res.shape}, min: {np.min(res)}, max: {np.max(res)}, mean: {np.mean(res)}, TQ: {Q3}')
    total = np.array(total_res)
    sort_counts_list = np.sort(total)
    Q3 = np.percentile(sort_counts_list, 75, method='midpoint')
    print(f'all: {total.shape}, mean: {np.mean(total)}, TQ: {Q3}')


estimate_positive_and_negative_samples(['my_mylyn'], step=1)
# estimate_positive_and_negative_samples(['my_mylyn'], step=2)
# estimate_positive_and_negative_samples(['my_mylyn'], step=3)
