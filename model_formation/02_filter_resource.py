# 对StructureKind == resource的event进行过滤
import os
from os.path import isdir, join
import xml.etree.ElementTree as ET
import xlwt

from data_count.time_util import get_common_time

write_excel = None
sheet1 = None
line_index = None
row0 = ["bug report", "Kind", "StructureKind", "StructureHandle", "StartDate"]


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_xsl():
    global write_excel
    global sheet1
    global line_index
    global row0
    line_index = 0
    write_excel = xlwt.Workbook()  # 创建工作表
    sheet1 = write_excel.add_sheet(u"Sheet1", cell_overwrite_ok=True)  # 创建sheet
    # 生成第一行
    for i in range(0, len(row0)):
        sheet1.write(line_index, i, row0[i])


def save_xsl(project_name: str, bug_id: str):
    global write_excel
    file_path = os.path.dirname(os.path.realpath(__file__))
    make_dir("resource/" + project_name)
    file_name = os.path.join(file_path, "resource/" + project_name + "/" + bug_id + ".xls")
    if os.path.exists(file_name):
        os.remove(file_name)
    # 保存文件
    write_excel.save(file_name)


def write_xsl(row: list):
    global line_index
    global sheet1
    global row0
    line_index += 1
    # 生成第一行
    for i in range(0, len(row0)):
        sheet1.write(line_index, i, row[i])


path = '../2023_dataset/mylyn_zip'

project_dir_list = os.listdir(path)

for project in project_dir_list:
    project_path = join(path, project)
    if not isdir(project_path):
        continue
    print('目录:', project, project_path)
    bug_dir_list = os.listdir(project_path)
    # 进入bug目录
    for bug in sorted(bug_dir_list, key=len):
        xml_counts = 0  # 当前bug的xml文件个数，也就是Interaction Traces个数
        bug_path = join(project_path, bug)
        if not isdir(bug_path):
            continue
        zip_dir_list = os.listdir(bug_path)
        # 读取到bug_zip
        for bug_zip in zip_dir_list:
            bug_zip_path = join(bug_path, bug_zip)
            if not isdir(bug_zip_path):
                continue
            xml_dir_list = os.listdir(bug_zip_path)
            bug_start_time_list = []
            # 开始读取xml文件
            for xml_file in xml_dir_list:
                if not xml_file.endswith('.xml'):
                    continue
                xml_counts += 1  # 增加traces个数
                if not xml_counts == len(zip_dir_list) / 2:  # 只要最后一个，如果不是直接跳过
                    continue
                init_xsl()
                xml_file_path = join(bug_zip_path, xml_file)
                tree = ET.parse(xml_file_path)
                # 拿到InteractionHistory节点
                root = tree.getroot()
                # 拿到所有的InteractionEvent
                events_nodes = root.findall('InteractionEvent')
                event_sum = 0
                for event in events_nodes:
                    # 161443 mylyn
                    event_kind = event.attrib.get('Kind')
                    event_structure_kind = event.attrib.get('StructureKind')
                    event_start_date = get_common_time(event.attrib.get('StartDate'))
                    event_structure_handle = event.attrib.get('StructureHandle')
                    if (event_kind == 'selection' or event_kind == 'edit') and event_structure_kind == 'resource' and (
                            event_structure_handle.endswith('.java') or event_structure_handle.find('.java[') > -1):
                        event_sum += 1
                        write_xsl([bug, event_kind, event_structure_kind, event_structure_handle, event_start_date])
                        print("{} bug {} event, StartDate is {}, StructureHandle is {}".format(bug, event_kind,
                                                                                               event_start_date,
                                                                                               event_structure_handle))
                if event_sum > 0:
                    save_xsl(project, bug)
