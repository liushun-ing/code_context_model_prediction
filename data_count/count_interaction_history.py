import os
from os.path import join, isdir

import numpy as np


path = '../bug_dataset/mylyn_zip'
project_dir_list = os.listdir(path)
print(project_dir_list)
total_count = []
for project in project_dir_list:
    count = []
    project_path = join(path, project)
    if not isdir(project_path) or project == 'ECF':
        continue
    print('目录:', project, project_path)
    bug_dir_list = os.listdir(project_path)
    bug_dir_list = sorted(bug_dir_list, key=lambda x: int(x))
    result = []
    # 进入bug目录
    for bug in bug_dir_list:
        # print('-----', bug)
        bug_path = join(project_path, bug)
        if not isdir(bug_path):
            continue
        zip_dir_list = os.listdir(bug_path)
        zip_dir_list = list(filter(lambda x: not x.endswith('.zip'), zip_dir_list))
        count.append(len(zip_dir_list))
        total_count.append(len(zip_dir_list))
    n = np.array(count)
    print(np.count_nonzero(n))
    print(n.size, np.mean(n), np.min(n), np.max(n), np.median(n), np.std(n))
n = np.array(total_count)
print(np.count_nonzero(n))
print(n.size, np.mean(n), np.min(n), np.max(n), np.median(n), np.std(n))
