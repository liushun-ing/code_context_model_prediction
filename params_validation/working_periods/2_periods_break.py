# 根据时间间隔对history进行分割，得到working periods
import os
import shutil
from datetime import datetime
from os.path import isdir, join
import xml.etree.ElementTree as ET

import pandas as pd
from data_count.time_util import get_common_time

PERIOD_INDEX = 0
TIME_ = 0
TIME_PERIOD = ''
TOTAL_DATA = []
XML_ROOT = None


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_xml(bug):
    global XML_ROOT
    XML_ROOT = ET.Element("bug_working_periods")
    XML_ROOT.set("bug_id", str(bug))


def save_xml(project_name: str):
    global XML_ROOT
    global PERIOD_INDEX
    file_path = os.path.dirname(os.path.realpath(__file__))
    make_dir(TIME_PERIOD + project_name)
    file_name = os.path.join(file_path, TIME_PERIOD + project_name + "/" + str(PERIOD_INDEX) + ".xml")
    if os.path.exists(file_name):
        os.remove(file_name)
    # 保存文件
    tree = ET.ElementTree(XML_ROOT)
    # 将XML写入文件
    tree.write(file_name)


def duration_is_over(date1: str, date2: str):
    time1 = datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')
    time2 = datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')
    duration = time2 - time1
    return (duration.total_seconds()) >= TIME_


def break_interaction_history(events_nodes: list[ET.Element], bug_id, project_name):
    global MODEL_ID
    global XML_ROOT
    global PERIOD_INDEX
    global TOTAL_DATA
    if len(events_nodes) <= 0:
        return
    transfer_nodes: list[dict] = []
    for node in events_nodes:
        transfer_nodes.append({
            "id": 0,
            "event_kind": node.attrib.get('Kind'),
            "event_structure_kind": node.attrib.get('StructureKind'),
            "event_start_date": get_common_time(node.attrib.get('StartDate')),
            "event_structure_handle": node.attrib.get('StructureHandle')
        })
    transfer_nodes.sort(key=lambda x: x.get('event_start_date'))  # 按照时间升序排列
    count = 0
    c_i, n_i = 0, 0  # current_index next_index
    init_xml(bug_id)
    PERIOD_INDEX += 1
    save_nodes = 0
    # 然后一个一个的判断
    while n_i < len(transfer_nodes):
        # 之前与前一个node的时间差达到时间间隔，就递增MODEL_ID,保存，初始化下一个
        if duration_is_over(transfer_nodes[c_i].get('event_start_date'), transfer_nodes[n_i].get('event_start_date')):
            save_xml(project_name)
            init_xml(bug_id)
            PERIOD_INDEX += 1
            count += 1
        en = transfer_nodes[n_i]
        event = ET.SubElement(XML_ROOT, 'event')
        event.set("event_kind", en.get("event_kind"))
        event.set("event_structure_kind", en.get("event_structure_kind"))
        event.set("event_structure_handle", en.get("event_structure_handle"))
        event.set("event_start_date", en.get("event_start_date"))
        save_nodes += 1
        c_i = n_i
        n_i += 1  # 进入下一个node
    save_xml(project_name)
    count += 1
    TOTAL_DATA.append((project_name, bug_id, count))
    print("{}'s {} bug has {} working periods, total/saved: {}/{}".format(project_name, bug_id, count,
                                                                          len(transfer_nodes), save_nodes))


def main_fun():
    global TOTAL_DATA
    global PERIOD_INDEX
    path = '../../2023_dataset/mylyn_zip'
    project_dir_list = os.listdir(path)
    TOTAL_DATA = []
    for project in project_dir_list:
        project_path = join(path, project)
        if not isdir(project_path):
            continue
        print('目录:', project, project_path)
        PERIOD_INDEX = 0
        bug_dir_list = os.listdir(project_path)
        bug_dir_list = sorted(bug_dir_list, key=lambda x: int(x))
        # 进入bug目录
        for bug in bug_dir_list:
            bug_path = join(project_path, bug)
            if not isdir(bug_path):
                continue
            zip_dir_list = os.listdir(bug_path)
            zip_dir_list = list(filter(lambda x: not x.endswith('.zip'), zip_dir_list))
            zip_dir_list.sort(key=lambda x: int(x.split('_')[1]))
            bug_zip_path = join(bug_path, zip_dir_list[-1])
            # 读取到最后一个traces的bug_zip
            if not isdir(bug_zip_path):
                continue
            xml_dir_list = os.listdir(bug_zip_path)
            # 开始读取xml文件
            for xml_file in xml_dir_list:
                if not xml_file.endswith('.xml'):
                    continue
                xml_file_path = join(bug_zip_path, xml_file)
                tree = ET.parse(xml_file_path)
                # 拿到InteractionHistory节点
                root = tree.getroot()
                # 拿到所有的InteractionEvent
                all_events_nodes = root.findall('InteractionEvent')
                break_interaction_history(all_events_nodes, bug, project)
    file_path = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(file_path, TIME_PERIOD, "total_working_data.tsv")
    if os.path.exists(file_name):
        os.remove(file_name)
    # 保存文件
    pd.DataFrame(TOTAL_DATA, columns=['project', 'bug_id', 'working_periods']).to_csv(file_name, index=False)


def process():
    global TIME_
    global TIME_PERIOD
    global PERIOD_INDEX
    times = [1800, 3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000]
    if os.path.exists('./periods'):
        shutil.rmtree('./periods')
    for i in range(10):
        TIME_ = times[i]
        TIME_PERIOD = 'periods/0' + str(i) + "/"
        PERIOD_INDEX = 0
        main_fun()


# process()
