"""
这个文件针对09文件计算生成的xml文件进行数据的统计分析
"""
import os
from os.path import join
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def show_model(context_models: list, model_info: str):
    model_info = 'model ' + model_info
    node_list = []
    edge_list = []
    component_list = []
    for model in context_models:
        node_list.append(int(model.get('nodes')))
        edge_list.append(int(model.get('edges')))
        component_list.append(int(model.get('components')))
    min_node = np.min(node_list)
    max_node = np.max(node_list)
    mean_node = np.mean(node_list)
    median_node = np.median(node_list)
    std_node = np.std(node_list)
    min_edge = np.min(edge_list)
    max_edge = np.max(edge_list)
    mean_edge = np.mean(edge_list)
    median_edge = np.median(edge_list)
    std_edge = np.std(edge_list)
    min_component = np.min(component_list)
    max_component = np.max(component_list)
    mean_component = np.mean(component_list)
    median_component = np.median(component_list)
    std_component = np.std(component_list)
    print("min_node:{0}, max_node:{1}, mean_node:{2}, median_node:{3}, std_node:{4}".format(min_node, max_node,
                                                                                            mean_node, median_node,
                                                                                            std_node))
    sns.displot(data=node_list)
    plt.title(model_info)
    plt.xlabel('node num')
    plt.ylabel('model count')
    plt.show()
    print("min_edge:{0}, max_edge:{1}, mean_edge:{2}, median_edge:{3}, std_edge:{4}".format(min_edge,
                                                                                            max_edge, mean_edge,
                                                                                            median_edge, std_edge))
    sns.displot(data=edge_list)
    plt.title(model_info)
    plt.xlabel('edge num')
    plt.ylabel('model count')
    plt.show()
    print(
        "min_component:{0}, max_component:{1}, mean_component:{2}, median_component:{3}, std_component:{4}".format(
            min_component,
            max_component,
            mean_component,
            median_component,
            std_component))
    sns.displot(data=component_list)
    plt.title(model_info)
    plt.xlabel('component num')
    plt.ylabel('model count')
    plt.show()


def show_component(connected_components: list, model_info: str):
    model_info = 'CC ' + model_info
    node_list = []
    edge_list = []
    diameter_list = []
    for cc in connected_components:
        node_list.append(int(cc.get('node_num')))
        edge_list.append(int(cc.get('edge_num')))
        diameter_list.append(int(cc.get('diameter')))
    min_node = np.min(node_list)
    max_node = np.max(node_list)
    mean_node = np.mean(node_list)
    median_node = np.median(node_list)
    std_node = np.std(node_list)
    min_edge = np.min(edge_list)
    max_edge = np.max(edge_list)
    mean_edge = np.mean(edge_list)
    median_edge = np.median(edge_list)
    std_edge = np.std(edge_list)
    min_diameter = np.min(diameter_list)
    max_diameter = np.max(diameter_list)
    mean_diameter = np.mean(diameter_list)
    median_diameter = np.median(diameter_list)
    std_diameter = np.std(diameter_list)
    print("min_node:{0}, max_node:{1}, mean_node:{2}, median_node:{3}, std_node:{4}".format(min_node, max_node,
                                                                                            mean_node, median_node,
                                                                                            std_node))
    sns.displot(data=node_list)
    plt.title(model_info)
    plt.xlabel('node num')
    plt.ylabel('CC count')
    plt.show()
    print("min_edge:{0}, max_edge:{1}, mean_edge:{2}, median_edge:{3}, std_edge:{4}".format(min_edge,
                                                                                            max_edge, mean_edge,
                                                                                            median_edge, std_edge))
    sns.displot(data=edge_list)
    plt.title(model_info)
    plt.xlabel('edge num')
    plt.ylabel('CC count')
    plt.show()
    print(
        "min_diameter:{0}, max_diameter:{1}, mean_diameter:{2}, median_diameter:{3}, std_diameter:{4}".format(
            min_diameter,
            max_diameter,
            mean_diameter,
            median_diameter,
            std_diameter))
    sns.displot(data=diameter_list)
    plt.title(model_info)
    plt.xlabel('diameter num')
    plt.ylabel('CC count')
    plt.show()


def main_func():
    info_dir_path = join(os.path.dirname(os.path.realpath(__file__)), 'model_info')
    model_info_list = os.listdir(info_dir_path)
    for model_info in model_info_list:
        print(model_info)
        model_file = join(info_dir_path, model_info)
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        context_models = root.findall('code_context_model')
        print("context model: ", len(context_models))
        show_model(context_models, model_info)
        connected_components = []
        for model in context_models:
            gs = model.findall('graph')
            for g in gs:
                connected_components += g.findall('connected_component')
        print("conneted components: ", len(connected_components))
        show_component(connected_components, model_info)


main_func()
