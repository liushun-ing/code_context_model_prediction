"""
07步骤的工具类
用于从项目中解析code elements所处的文件夹
"""
import os
from os.path import isdir, join


def filter_dire(path: str):
    """
    过滤有效的dir 不已.开头，并且本身是目录

    :param path: 目录
    :return: 返回该目录下的有效 dir list
    """
    file_dir_list = os.listdir(path)
    return list(filter(lambda dir_str: not dir_str.startswith('.') and isdir(join(path, dir_str)), file_dir_list))


def extract_repos(elements: list[str]):
    """
    从code elements中解析出repo

    :param elements: code elements
    :return: 不重复的repo list
    """
    repos = set()
    for element in elements:
        index = element.find('/src&lt;')
        if index == -1:
            continue
        repos.add(element[1: index])
    return list(repos)


def find_repo_deep(path: str, repo: str):
    """
    递归查找是否存在某个项目，如果某个目录下含有src目录，则可以结束递归

    :return: 如果找到了，返回项目路径
    """
    sub_dir_list = filter_dire(path)
    # print("finding in {0} has dir: {1}".format(path, sub_dir_list))
    if sub_dir_list.count('src') > 0:
        return ''
    for sub_dir in sub_dir_list:
        if sub_dir == repo:
            return join(path, sub_dir)
        else:
            res = find_repo_deep(join(path, sub_dir), repo)
            if res:
                return res


def solve_element_directory(path: str, elements: list[str]):
    """
    根据项目path和code elements查找code elements所在的目录

    :param path: 项目路径
    :param elements: code elements
    :return: 找到的项目路径 list
    """
    repos = extract_repos(elements)
    print("repos: {0}".format(repos))
    repo_path_list = []
    for repo in repos:
        # print("now finding {0}".format(repo))
        repo_path = find_repo_deep(path, repo)
        if repo_path:
            repo_path_list.append(join(repo_path, 'src'))
        # print("found result: {0}".format(repo_path))
    print("found repo path results:{0}".format(repo_path_list))
    return repo_path_list
