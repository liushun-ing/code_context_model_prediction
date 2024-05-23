"""
收集每个 bug duration数据
"""
import ast
import os
from os.path import isdir, join
import xml.etree.ElementTree as ET
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_count.time_util import get_common_time


path = '../../bug_dataset/mylyn_zip'


def get_duration_of_event(date_list: list[str]):
    duration_list = []  # 相差的秒数
    for i in range(len(date_list) - 1):
        time1 = datetime.strptime(date_list[i], '%Y-%m-%d %H:%M:%S')
        time2 = datetime.strptime(date_list[i + 1], '%Y-%m-%d %H:%M:%S')
        duration = time2 - time1
        duration_list.append(duration.total_seconds())
    return duration_list


def save_to_tsv():
    project_dir_list = os.listdir(path)
    for project in project_dir_list:
        project_path = join(path, project)
        if not isdir(project_path):
            continue
        print('目录:', project, project_path)
        bug_dir_list = os.listdir(project_path)
        bug_dir_list = sorted(bug_dir_list, key=lambda x: int(x))
        result = []
        # 进入bug目录
        for bug in bug_dir_list:
            print('-----', bug)
            duration_list = []
            bug_path = join(project_path, bug)
            if not isdir(bug_path):
                continue
            zip_dir_list = os.listdir(bug_path)
            print(zip_dir_list)
            zip_dir_list = list(filter(lambda x: not x.endswith('.zip'), zip_dir_list))
            zip_dir_list.sort(key=lambda x: int(x.split('_')[1]))
            print(zip_dir_list)
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
                events_nodes = root.findall('InteractionEvent')
                # 拿到所有event的开始时间
                start_date_list = []
                for event in events_nodes:
                    start_date_list.append(event.attrib.get('StartDate'))
                for i in range(len(start_date_list)):
                    start_date_list[i] = get_common_time(start_date_list[i])
                start_date_list.sort()
                duration_list = get_duration_of_event(start_date_list)  # duration列表
            # print(duration_list)
            result.append((bug, duration_list))
        df = pd.DataFrame(result, columns=['bug_id', 'durations'])
        df.to_csv(f'./{project}.tsv', index=False)


def process():
    project_dir_list = os.listdir(path)
    # 计算比例
    times = [1800, 3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000]
    times_ = ['0.5h', '1h', '1.5h', '2h', '2.5h', '3h', '3.5h', '4h', '4.5h', '5h']
    final_avg = []
    final_quarter = []
    final_max = []
    final_all = []
    d = np.array([])
    for project in project_dir_list:
        print(f'-----{project}-----')
        df = pd.read_csv(f'./{project}.tsv')
        df['durations'] = df['durations'].apply(lambda x: np.array(ast.literal_eval(x)))
        df.insert(loc=2, column='average', value=df['durations'].apply(lambda x: np.mean(x)).tolist())
        df.insert(loc=3, column='quarter', value=df['durations'].apply(lambda x: np.percentile(np.sort(x), 98, method='midpoint')).tolist())
        df.insert(loc=4, column='max', value=df['durations'].apply(lambda x: np.max(x)).tolist())
        avg_ = []
        quarter_ = []
        max_ = []
        all_ = []
        for time_period in times:
            avg_percent = np.count_nonzero(df['average'] < time_period) / len(df['average'])
            quarter_percent = np.count_nonzero(df['quarter'] < time_period) / len(df['quarter'])
            max_percent = np.count_nonzero(df['max'] < time_period) / len(df['max'])
            all_duration = np.concatenate(df['durations'].tolist())
            d = np.concatenate([d, all_duration])
            all_percent = np.count_nonzero(all_duration < time_period) / len(all_duration)
            avg_.append(avg_percent)
            quarter_.append(quarter_percent)
            max_.append(max_percent)
            all_.append(all_percent)
            # print(f'avg: {avg_percent}, quarter: {quarter_percent}, max: {max_percent}, all: {all_percent}')
        final_avg.append(avg_)
        final_quarter.append(quarter_)
        final_max.append(max_)
        final_all.append(all_)
        print(np.mean(all_duration))
    print(np.mean(d))

    plt.plot(times_, final_avg[0], '.-', label='Platform')
    plt.plot(times_, final_avg[1], '.-', label='ECF')
    plt.plot(times_, final_avg[2], '.-', label='PDE')
    plt.plot(times_, final_avg[3], '.-', label='Mylyn')
    # plt.title('InteractionHistory whose average duration below time threshold')
    plt.xlabel('time threshold')
    plt.ylabel('InteractionHistory proportion')
    plt.legend()
    plt.show()
    plt.plot(times_, final_max[0], '.-', label='Platform')
    plt.plot(times_, final_max[1], '.-', label='ECF')
    plt.plot(times_, final_max[2], '.-', label='PDE')
    plt.plot(times_, final_max[3], '.-', label='Mylyn')
    # plt.title('max InteractionHistory duration below time threshold')
    plt.xlabel('time threshold')
    plt.ylabel('InteractionHistory proportion')
    plt.legend()
    plt.show()
    plt.plot(times_, final_all[0], '.-', label='Platform')
    plt.plot(times_, final_all[1], '.-', label='ECF')
    plt.plot(times_, final_all[2], '.-', label='PDE')
    plt.plot(times_, final_all[3], '.-', label='Mylyn')
    # plt.title('durations below time threshold')
    plt.xlabel('time threshold')
    plt.ylabel('duration proportion')
    plt.legend()
    plt.show()


# save_to_tsv()
process()
