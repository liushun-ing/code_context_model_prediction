"""
07步骤工具类
根据solve_file工具类得到的path路径
去解析code elements对应的文件，将文件copy到另一个目录中，减少doxygen解析量
"""
import os
import shutil
from os.path import join, exists


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def extract_element_java_file(repo_list: list[str], elements: list[str]):
    """
    根据repo_list进行分类，提取各个项目的code elements所对应的java文件
    :param repo_list: 项目列表
    :param elements: code elements
    :return: {repo: set[java_file_path]}
    """
    result: dict[str, set[str]] = dict()
    count = 0
    for repo in repo_list:
        src_index = repo.find("/src")
        repo_index = repo.rfind("/", 0, src_index)
        result[repo[repo_index + 1: src_index]] = set()
    for element in elements:
        index1 = element.find("/src&lt;")
        index2 = element.find("{")
        index3 = element.rfind(".java")
        if index1 == -1 or index2 == -1 or index3 == -1:
            continue
        project_path = element[1: index1]
        package_path = element[index1 + 8: index2].replace(".", "/")
        file_path = element[index2 + 1: index3 + 5]
        for repo in repo_list:
            if repo.find(project_path) > 0:
                java_file = join(repo, package_path, file_path)
                if exists(java_file):
                    count += 1
                    result.get(project_path).add(java_file)
    print('{} elements find {} file'.format(len(elements), count))
    return result


def copy_new_file(root: str, java_file_map: dict[str, set[str]]):
    """
    复制相关文件到root目录
    :param root: 目标目录
    :param java_file_map: 解析出来java文件路径 {repo: set[java_file_path]}
    :return: copy项目保存src路径
    """
    result = []
    for item in java_file_map.items():
        repo = item[0]
        java_set = item[1]
        result.append(join(root, repo, 'src'))
        for java_file in java_set:
            start_index = java_file.find(repo)
            end_index = java_file.rfind('/')
            new_dir = java_file[start_index: end_index]
            java_name = java_file[end_index + 1:]
            make_dir(join(root, new_dir))
            new_file_path = join(root, new_dir, java_name)
            shutil.copy(java_file, new_file_path)
    return result


def extract_element_project_repos(repo_list: list[str], elements: list[str]):
    """
    根据repo_list进行分类，提取各个项目的资源跟目录
    :param repo_list: 项目列表
    :param elements: code elements
    :return: [(repo: project_root_path)]
    """
    result: dict[str, str] = dict()
    for repo in repo_list:  # 解析出根目录，去除src层
        src_index = repo.find("/src")
        repo_index = repo.rfind("/", 0, src_index)
        result[repo[repo_index + 1: src_index]] = repo
    print('{} elements find {} project repo'.format(len(elements), len(result)))
    return result


def copy_new_project(root: str, java_projects: dict[str, str]):
    """
    复制相关文件到 root 目录

    :param root: 目标目录
    :param java_projects: 解析出来java文件路径 {repo: set[java_file_path]}
    :return: copy项目保存src路径
    """
    result = []
    for project_dict in java_projects.items():
        project_repo = project_dict[0]
        old_project_path = project_dict[1]
        new_project_repo_path = join(root, project_repo)
        print('1111', old_project_path, os.path.exists(old_project_path))
        if os.path.exists(new_project_repo_path):
            # 如果目标路径存在原文件夹的话就先删除
            shutil.rmtree(new_project_repo_path)
        # make_dir(new_project_repo_path)
        new_project_path = join(root, project_repo, 'src')
        result.append(new_project_path)
        if os.path.exists(new_project_path):
            # 如果目标路径存在原文件夹的话就先删除
            shutil.rmtree(new_project_path)
        shutil.copytree(old_project_path, new_project_path)
    return result


def copy_element_file(root: str, repo_list: list[str], elements: list[str]):
    """
    提取code elements关联得文件（修改为提取code elements相关的项目），方便后续运行doxygen

    :param root: 保存文件的根目录
    :param repo_list: 解析出来的repo仓库路径
    :param elements: code elements
    :return:
    """
    # print("root: {0}, repo list: {1}, elements: {2}".format(root, repo_list, elements))
    # 解析出project,这里直接解析出项目，是为了后续进行扩展得到大图
    java_projects = extract_element_project_repos(repo_list, elements)
    print("java projects : {0}".format(java_projects))
    # 将项目文件copy到root目录
    result = copy_new_project(root, java_projects)
    print("copy projects finished")
    return result

