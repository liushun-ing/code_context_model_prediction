"""
这里会根据计算出来的分位线来过滤异常值
保存到 IQR_code_timestamp 中
"""
import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET


period_index = 0


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_xml_excel(url):
    """
    参数：
        url:文件路径
    """
    tree = ET.parse(url)  # 拿到xml树
    # 获取XML文档的根元素
    root = tree.getroot()
    events = root.find('code_elements')
    return events.get('total')


def main_func(dir_list: list[str], outliers: dict()):
    file_path = join(os.path.dirname(os.path.realpath(__file__)), 'code_timestamp')
    file_dir_list = os.listdir(file_path)
    global period_index
    period_index = 0
    for index_dir in file_dir_list:
        if index_dir not in dir_list:
            continue
        index_path = join(file_path, index_dir)
        project_list = os.listdir(index_path)
        # 进入项目目录
        for project_dir in project_list:
            if project_dir not in ['PDE', 'Mylyn', 'Platform', 'ECF']:
                continue
            make_dir("IQR_code_timestamp/" + index_dir + '/' + project_dir)
            print("current project: {0}".format(project_dir))
            project_path = join(index_path, project_dir)
            xml_list = os.listdir(project_path)
            xml_list = [x[:x.find('.')] for x in xml_list]
            xml_list = sorted(xml_list, key=lambda x: int(x))
            xml_list = [x + ".xml" for x in xml_list]
            for xml_file in xml_list:
                xml_path = join(project_path, xml_file)
                total_event = read_xml_excel(xml_path)  # 拿到所有的data
                if int(total_event) > outliers.get(project_dir)[1]:
                    continue
                period_index += 1
                new_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        "IQR_code_timestamp/" + index_dir + '/' + project_dir + '/' + str(
                                            period_index) + ".xml")
                shutil.copy(xml_path, new_path)
                print('success copy {0} {1} {2} to {3} {4} {5}.xml'.format(
                    index_dir, project_dir, xml_file, index_dir, project_dir, period_index
                ))


# 2h
main_func(['04'], {
    "ECF": [-28.5, 45],
    "Mylyn": [-67, 101],
    "PDE": [-31, 46],
    "Platform": [-110, 163]
})
# 3h
main_func(['06'], {
    "ECF": [-30, 47],
    "Mylyn": [-70, 105],
    "PDE": [-31, 46],
    "Platform": [-112, 168]
})
