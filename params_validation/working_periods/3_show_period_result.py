import os

import pandas as pd
from matplotlib import pyplot as plt


def count_working_periods():
    times_ = ['0.5h', '1h', '1.5h', '2h', '2.5h', '3h', '3.5h', '4h', '4.5h', '5h']
    mylyn, platform, ecf, pde = [], [], [], []
    for i in range(10):
        file_path = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(file_path, 'periods/0' + str(i) + "/", "total_working_data.tsv")
        df = pd.read_csv(file_name)  # columns=['project', 'bug_id', 'working_periods']
        mylyn_df = df[df['project'] == 'Mylyn']
        platform_df = df[df['project'] == 'Platform']
        ecf_df = df[df['project'] == 'ECF']
        pde_df = df[df['project'] == 'PDE']
        m_bug, m_period = mylyn_df['bug_id'].count(), mylyn_df['working_periods'].sum()
        p_bug, p_period = platform_df['bug_id'].count(), platform_df['working_periods'].sum()
        e_bug, e_period = ecf_df['bug_id'].count(), ecf_df['working_periods'].sum()
        pde_bug, pde_period = pde_df['bug_id'].count(), pde_df['working_periods'].sum()
        mylyn.append(m_period)
        platform.append(p_period)
        ecf.append(e_period)
        pde.append(pde_period)
        print(f'{times_[i]} bugs: mylyn:{m_bug}, platform:{p_bug}, ecf:{e_bug}, pde:{pde_bug}')
        print(f'{times_[i]} periods: mylyn:{m_period}, platform:{p_period}, ecf:{e_period}, pde:{pde_period}')
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
    plt.plot(times_, ecf, '.-')
    plt.title('ECF')
    plt.xlabel('time threshold')
    plt.ylabel('working periods')
    for i in range(10):
        plt.text(times_[i], ecf[i], ecf[i], va='bottom')
    plt.show()
    plt.plot(times_, pde, '.-')
    plt.title('PDE')
    plt.xlabel('time threshold')
    plt.ylabel('working periods')
    for i in range(10):
        plt.text(times_[i], pde[i], pde[i], va='bottom')
    plt.show()
    plt.plot(times_, mylyn, '.-', label='Mylyn')
    plt.plot(times_, platform, '.-', label='Platform')
    plt.plot(times_, pde, '.-', label='PDE')
    plt.plot(times_, ecf, '.-', label='ECF')
    # plt.title('working periods')
    plt.xlabel('time threshold')
    plt.ylabel('working periods number')
    plt.legend()
    plt.show()


count_working_periods()
