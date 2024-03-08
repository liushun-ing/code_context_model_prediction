# 提取code element和最早最晚的startDate
# 所有的无效过滤条件，不需要过滤重复的code elements
# 最终留下来的就是有效的working periods了
import os
from os.path import join
import numpy as np
import xlwt
from collections import Counter
import xml.etree.ElementTree as ET

xml_root = None
period_index = 0


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_xml(bug_id):
    global xml_root
    xml_root = ET.Element("bug_filter_working_periods")
    xml_root.set("bug_id", str(bug_id))


def save_xml(xml_file_path):
    global xml_root
    if os.path.exists(xml_file_path):
        os.remove(xml_file_path)
    # 保存文件
    tree = ET.ElementTree(xml_root)
    # 将XML写入文件
    tree.write(xml_file_path)


def read_xml_excel(url):
    """
    读取xml格式文件
    参数：
        url:文件路径
    返回：
        data:表格中的数据
    """
    tree = ET.parse(url)  # 拿到xml树
    # 获取XML文档的根元素
    root = tree.getroot()
    events = root.findall('event')
    data = []
    # 遍历每一行数据
    for event in events:
        # 定义表格存储每一行数据
        da = [event.get("event_kind"), event.get("event_structure_kind"),
              event.get("event_structure_handle"), event.get("event_start_date")]
        # 存储每一行数据
        data.append(da)
    # 返回数据
    return str(root.get('bug_id')), data


def main_func():
    file_path = join(os.path.dirname(os.path.realpath(__file__)), 'periods')
    file_dir_list = os.listdir(file_path)
    global xml_root
    global period_index
    for index_dir in file_dir_list:
        if index_dir not in ['01', '02', '03', '04', '05', '06', '07', '08']:
            continue
        period_index = 0
        index_path = join(file_path, index_dir)
        project_list = os.listdir(index_path)
        total_data_work_excel_1 = xlwt.Workbook()  # 创建工作表
        # 进入项目目录
        for project_dir in project_list:
            if project_dir not in ['PDE', 'Mylyn', 'Platform', 'ECF']:
                continue
            make_dir("code_elements/" + index_dir + '/' + project_dir)
            project_path = join(index_path, project_dir)
            xml_list = os.listdir(project_path)
            # 用来统计每个working period的event个数信息
            project_sheet_1 = total_data_work_excel_1.add_sheet(project_dir, cell_overwrite_ok=True)  # 创建sheet
            project_line_index_1 = 0
            project_sheet_1.write(project_line_index_1, 0, "bug")
            project_sheet_1.write(project_line_index_1, 1, "working period id")
            project_sheet_1.write(project_line_index_1, 2, "contain events")
            xml_list = [x[:x.find('.')] for x in xml_list]
            xml_list = sorted(xml_list, key=lambda x: int(x))
            xml_list = [x + ".xml" for x in xml_list]
            for period_xml_file in xml_list:
                xml_path = join(project_path, period_xml_file)
                bug_id, xml_data = read_xml_excel(xml_path)  # 拿到所有的data
                # 过滤无效的event,并记录最早和最晚的startDate
                a = np.array(xml_data)
                # 提取第一个和最后一个event的startDate
                all_start_date = a[:, -1]
                first_date, last_date = min(all_start_date), max(all_start_date)
                con = []
                # 只要有效，可以重复，方便后续提取时间
                for line in xml_data:
                    # 有效才返回True,留下来
                    e_kind = line[0]
                    e_structure_king = line[1]
                    if ((e_kind == 'selection' or e_kind == 'edit') and e_structure_king == 'java') and (
                            line[2].endswith('.java') or line[2].find('.java[') > -1):
                        con.append(True)
                    else:
                        con.append(False)
                effective_data = a[con]  # 得到有效的event
                # 写入新的文件
                if len(effective_data) <= 0:
                    continue
                else:
                    period_index += 1
                    init_xml(bug_id)
                    xml_root.set("total", str(len(effective_data)))
                    time_stamp = ET.SubElement(xml_root, "timestamp")
                    time_stamp.set("start", first_date)
                    time_stamp.set("last", last_date)
                    for en in effective_data:
                        event = ET.SubElement(xml_root, 'event')
                        event.set("event_kind", en[0])
                        event.set("event_structure_kind", en[1])
                        event.set("event_structure_handle", en[2])
                        event.set("event_start_date", en[3])
                    xls_file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 "code_elements/" + index_dir + '/' + project_dir + "/" + str(
                                                     period_index) + ".xml")
                    save_xml(xls_file_name)
                    print("handle {0} project {1}'s {2} xml".format(index_dir, project_dir, period_index))
                    # 统计每个working period的event个数信息
                    all_ids = effective_data[:, 0]
                    id_groups = Counter(all_ids)
                    for key_id in id_groups.keys():
                        project_line_index_1 += 1
                        project_sheet_1.write(project_line_index_1, 0, period_xml_file)
                        project_sheet_1.write(project_line_index_1, 1, key_id)
                        project_sheet_1.write(project_line_index_1, 2, id_groups.get(key_id))
        # 保存event统计文件
        total_file_name_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "code_elements/" + index_dir + "/working_periods_event_count.xls")
        if os.path.exists(total_file_name_1):
            os.remove(total_file_name_1)
        total_data_work_excel_1.save(total_file_name_1)


main_func()
