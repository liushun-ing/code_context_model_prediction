# 该文件用于提取所有bug_zip中的xml文件到相应的目录中
import os
import shutil
from os.path import isfile, isdir, join
import zipfile


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


path = '../bug_dataset/mylyn_zip'

project_dir_list = os.listdir(path)

for project in project_dir_list:
    project_path = join(path, project)
    if not isdir(project_path):
        continue
    print('目录:', project, project_path)
    bug_dir_list = os.listdir(project_path)
    # 进入bug目录
    for bug in bug_dir_list:
        bug_path = join(project_path, bug)
        if not isdir(bug_path):
            continue
        zip_dir_list = os.listdir(bug_path)
        # 读取到bug_zip
        for bug_zip in zip_dir_list:
            bug_zip_path = join(bug_path, bug_zip)
            print(bug_zip_path)
            if not zipfile.is_zipfile(bug_zip_path):
                continue
            zip_file = zipfile.ZipFile(bug_zip_path, 'r')
            # 读取到zip里的xml文件
            copy_path = bug_zip_path.replace(".zip", "_zip")
            make_dir(copy_path)
            for xml_file in zip_file.namelist():
                zip_file.extract(xml_file, copy_path)
            zip_file.close()

