# 该文件用于统计每个bug开始时间和结束时间和InteractionEvent个数等数据
import os
from os.path import isdir, join
import xml.etree.ElementTree as ET
from datetime import datetime
import xlwt

from data_count.time_util import get_common_time

write_excel = None
sheet1 = None
line_index = None
row0 = ["bug report", "Interaction Traces", "xml", "InteractionEvent", "min StartDate", "max StartDate",
        "event_duration(/s)", "bug_min_date", "bug_max_date"]


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


def save_xsl(project_name: str):
    global write_excel
    file_name = project_name + ".xls"
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


path = '../bug_dataset/mylyn_zip'

project_dir_list = os.listdir(path)

total_bugs = 0
total_traces = 0
total_events = 0


def get_duration_of_event(date_list: list[str]):
    duration_list = []  # 相差的秒数
    for i in range(len(date_list) - 1):
        time1 = datetime.strptime(date_list[i], '%Y-%m-%d %H:%M:%S')
        time2 = datetime.strptime(date_list[i + 1], '%Y-%m-%d %H:%M:%S')
        duration = time2 - time1
        duration_list.append(duration.total_seconds())
    return 0 if len(duration_list) == 1 else sum(duration_list) / (len(duration_list) - 1)
    # 这里基数应该还要减一，不过不影响了，这里的这个数据不重要，他又变得重要了，我哭死


for project in project_dir_list:
    init_xsl()
    total_bugs = 0
    total_traces = 0
    total_events = 0
    project_path = join(path, project)
    if not isdir(project_path):
        continue
    print('目录:', project, project_path)
    bug_dir_list = os.listdir(project_path)
    total_bugs = len(bug_dir_list)
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
                xml_file_path = join(bug_zip_path, xml_file)
                tree = ET.parse(xml_file_path)
                # 拿到InteractionHistory节点
                root = tree.getroot()
                # 拿到所有的InteractionEvent
                events_nodes = root.findall('InteractionEvent')
                event_counts = len(events_nodes)  # xml文件中的InteractionEvent个数
                total_events += event_counts
                # print(events_nodes)
                # 拿到所有event的开始时间
                start_date_list = []
                for event in events_nodes:
                    start_date_list.append(event.attrib.get('StartDate'))
                for i in range(len(start_date_list)):
                    start_date_list[i] = get_common_time(start_date_list[i])
                start_date_list.sort()
                average_duration = get_duration_of_event(start_date_list)  # duration平均值
                # print(start_date_list)
                # 统一为UTC时间
                min_start_date = start_date_list[0]  # 最早开始时间
                max_start_date = start_date_list[len(start_date_list) - 1]  # 最晚开始时间
                bug_start_time_list.append(min_start_date)
                bug_start_time_list.append(max_start_date)
                print("bug {} has {} Interaction Traces, xml {} has {} events, min StartDate is {}, max StartDate is {}"
                      .format(bug, xml_counts, bug_zip, event_counts, min_start_date, max_start_date))
                if xml_counts == len(zip_dir_list) / 2:
                    total_traces += xml_counts
                    write_xsl(
                        [bug, xml_counts, bug_zip, event_counts, min_start_date, max_start_date,
                         average_duration, min(bug_start_time_list), max(bug_start_time_list)])
                else:
                    write_xsl(
                        ['', xml_counts, bug_zip, event_counts, min_start_date, max_start_date, average_duration, '',
                         ''])
    write_xsl([total_bugs, total_traces, '', total_events, '', '', '', '', ''])
    save_xsl(project)
