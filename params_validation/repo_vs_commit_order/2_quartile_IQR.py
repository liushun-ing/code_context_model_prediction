"""
此文件用于计算各个项目的code elements的四分位数，检测数据异常
需要将输出的文件复制到 IQR.txt 中
"""
# 提取code element和最早最晚的startDate
import os
from os.path import join
import numpy as np
import xml.etree.ElementTree as ET


def read_xml_excel(url):
    """
    参数：
        url:文件路径
    """
    tree = ET.parse(url)  # 拿到xml树
    # 获取XML文档的根元素
    root = tree.getroot()
    events = root.find('code_elements')
    return int(events.get('total'))


def main_func(dir_list: list[str]):
    file_path = join(os.path.dirname(os.path.realpath(__file__)), 'code_timestamp')
    file_dir_list = os.listdir(file_path)
    for index_dir in file_dir_list:
        # 进入  code_elements/06 文件夹 只考虑3h的
        if index_dir not in dir_list:
            continue
        print(index_dir, '-----------')
        index_path = join(file_path, index_dir)
        project_list = os.listdir(index_path)
        # 进入项目目录
        for project_dir in project_list:
            if project_dir not in ['PDE', 'Mylyn', 'Platform', 'ECF']:
                continue
            print("current project: {0}".format(project_dir))
            project_path = join(index_path, project_dir)
            xml_list = os.listdir(project_path)
            xml_list = [x[:x.find('.')] for x in xml_list]
            xml_list = sorted(xml_list, key=lambda x: int(x))
            xml_list = [x + ".xml" for x in xml_list]
            events_counts_list = []  # 记录所有event个数信息
            # 收集所有working periods的event个数信息
            for xml_file in xml_list:
                xml_path = join(project_path, xml_file)
                total_event = read_xml_excel(xml_path)  # 拿到所有的data
                events_counts_list.append(total_event)
            sort_counts_list = np.sort(events_counts_list)
            Q1 = np.percentile(sort_counts_list, 25, method='midpoint')
            Q3 = np.percentile(sort_counts_list, 75, method='midpoint')
            print('Q1 = ', Q1)
            print('Q3 = ', Q3)
            IQR = Q3 - Q1
            print('IQR = ', IQR)
            low_lim = Q1 - 3 * IQR
            up_lim = Q3 + 3 * IQR
            print('Q1 - 3 * IQR = ', low_lim)
            print('Q3 + 3 * IQR = ', up_lim)
            print("----------------------------------------------")


# 3h
main_func(['05'])
