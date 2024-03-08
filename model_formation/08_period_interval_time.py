"""
用来统计每个working periods的时间间隔大小
"""
import os
from os.path import join
from datetime import datetime
import xml.etree.ElementTree as ET

import xlwt


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_duration(date1: str, date2: str):
    time1 = datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')
    time2 = datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')
    duration = time2 - time1
    return duration.seconds + duration.days * 86400


def main_func():
    file_path = join(os.path.dirname(os.path.realpath(__file__)), 'IQR_code_timestamp')
    file_dir_list = os.listdir(file_path)
    # 读取working periods
    for index_dir in file_dir_list:
        # 进入06文件夹
        if index_dir not in ['04', '06']:
            continue
        print('---------------', index_dir)
        index_path = join(file_path, index_dir)
        project_list = os.listdir(index_path)
        duration_excel = xlwt.Workbook()  # 创建工作表
        # 进入项目目录
        for project_dir in project_list:
            if project_dir not in ['ECF', 'Mylyn', 'PDE', 'Platform']:
                continue
            project_path = join(index_path, project_dir)
            xml_list = os.listdir(project_path)
            make_dir("period_duration/" + index_dir)
            # 用来统计每个working period的event个数信息
            project_sheet = duration_excel.add_sheet(project_dir, cell_overwrite_ok=True)  # 创建sheet
            project_line_index = 0
            project_sheet.write(project_line_index, 0, "working periods")
            project_sheet.write(project_line_index, 1, "first time")
            project_sheet.write(project_line_index, 2, "last time")
            project_sheet.write(project_line_index, 3, 'duration')
            xml_list = [x[:x.find('.')] for x in xml_list]
            xml_list = sorted(xml_list, key=len)
            xml_list = [x+".xml" for x in xml_list]
            for xml_file in xml_list:
                tree = ET.parse(join(project_path, xml_file))  # 拿到xml树
                # 获取XML文档的根元素
                root = tree.getroot()
                timestamps = root.find("timestamps")
                first_time = timestamps.find('first').text
                last_time = timestamps.find("last").text
                duration = get_duration(first_time, last_time)
                project_line_index += 1
                project_sheet.write(project_line_index, 0, xml_file)
                project_sheet.write(project_line_index, 1, first_time)
                project_sheet.write(project_line_index, 2, last_time)
                project_sheet.write(project_line_index, 3, duration)
                print("working period {0} duration is {1}".format(xml_file, duration))
        # 保存event统计文件
        file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "period_duration/" + index_dir + "/periods_duration_count.xls")
        if os.path.exists(file_name):
            os.remove(file_name)
        duration_excel.save(file_name)


main_func()
