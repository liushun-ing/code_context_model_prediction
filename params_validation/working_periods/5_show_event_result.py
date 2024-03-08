import os
from os.path import join

import pandas as pd
from matplotlib import pyplot as plt


def draw_periods(bugs_result):
    [mylyn, platform, pde, ecf] = bugs_result
    times_ = ['0.5h', '1h', '1.5h', '2h', '2.5h', '3h', '3.5h', '4h', '4.5h', '5h']
    plt.plot(times_, mylyn, '.-')
    plt.title('Mylyn')
    plt.xlabel('time threshold')
    plt.ylabel('working periods')
    for i in range(10):
        plt.text(times_[i], mylyn[i], mylyn[i], va='bottom')
    plt.show()
    plt.plot(times_, platform, '.-')
    plt.title('Platform')
    plt.xlabel('time threshold')
    plt.ylabel('working periods')
    for i in range(10):
        plt.text(times_[i], platform[i], platform[i], va='bottom')
    plt.show()
    plt.plot(times_, pde, '.-')
    plt.title('PDE')
    plt.xlabel('time threshold')
    plt.ylabel('working periods')
    for i in range(10):
        plt.text(times_[i], pde[i], pde[i], va='bottom')
    plt.show()
    plt.plot(times_, ecf, '.-')
    plt.title('ECF')
    plt.xlabel('time threshold')
    plt.ylabel('working periods')
    for i in range(10):
        plt.text(times_[i], ecf[i], ecf[i], va='bottom')
    plt.show()
    plt.plot(times_, mylyn, '.-', label='Mylyn')
    plt.plot(times_, platform, '.-', label='Platform')
    plt.plot(times_, pde, '.-', label='PDE')
    plt.plot(times_, ecf, '.-', label='ECF')
    plt.title('valid working periods')
    plt.xlabel('time threshold')
    plt.ylabel('periods number')
    plt.legend()
    plt.show()


def draw_events(events_result):
    [mylyn, platform, pde, ecf] = events_result
    times_ = ['0.5h', '1h', '1.5h', '2h', '2.5h', '3h', '3.5h', '4h', '4.5h', '5h']
    plt.plot(times_, mylyn, '.-', label='Mylyn')
    plt.plot(times_, platform, '.-', label='Platform')
    plt.plot(times_, pde, '.-', label='PDE')
    plt.plot(times_, ecf, '.-', label='ECF')
    # plt.title('average events of periods')
    plt.xlabel('time threshold')
    plt.ylabel('average interaction event number')
    plt.legend()
    plt.show()


def process():
    file_path = join(os.path.dirname(os.path.realpath(__file__)), 'code_elements')
    file_dir_list = os.listdir(file_path)
    bugs_result = [[], [], [], []]
    events_result = [[], [], [], []]
    for index_dir in sorted(file_dir_list, key=lambda x: int(x)):
        if index_dir not in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']:
            continue
        total_file_name = os.path.join(file_path, index_dir, "working_periods_events.tsv")
        df = pd.read_csv(total_file_name)  # columns=['project', 'bug_id', 'period_index', 'events']
        mylyn = df[df['project'] == 'Mylyn']
        platform = df[df['project'] == 'Platform']
        pde = df[df['project'] == 'PDE']
        ecf = df[df['project'] == 'ECF']
        # 统计 working periods 信息
        bugs_result[0].append(mylyn['period_index'].count())
        bugs_result[1].append(platform['period_index'].count())
        bugs_result[2].append(pde['period_index'].count())
        bugs_result[3].append(ecf['period_index'].count())
        # 统计平均 events
        events_result[0].append(int(mylyn['events'].mean()))
        # print(mylyn['events'].max())
        events_result[1].append(int(platform['events'].mean()))
        events_result[2].append(int(pde['events'].mean()))
        events_result[3].append(int(ecf['events'].mean()))

    # draw_periods(bugs_result)
    draw_events(events_result)


process()
