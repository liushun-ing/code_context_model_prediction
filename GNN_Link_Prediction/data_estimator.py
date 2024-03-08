"""
用于做一些数据的评估工作
"""
import os
from os.path import join
import xml.etree.ElementTree as ET
import numpy as np

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')


def estimate_positive_and_negative_samples(project_model_name: str):
    project_path = join(root_path, project_model_name)
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    ratios = []
    for model_dir in model_dir_list[0:500]:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, '1_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        for graph in graphs:
            count = 0
            edges = graph.find('edges')
            edge_list = edges.findall('edge')
            total = len(edge_list)
            for edge in edge_list:
                if edge.get('origin') == '1':
                    count += 1
            if total > 0:
                ratios.append((total - count) / (count if count > 0 else 1))
    res = np.array(ratios)
    print(f'ratios: {ratios}, mean: {np.mean(res)}, median: {np.median(res)}')


estimate_positive_and_negative_samples('my_mylyn')
