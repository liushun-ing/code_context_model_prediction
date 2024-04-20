"""
07步骤工具类
用于运行doxygen，输入项目路径，输出doxygen分析xml文件
"""
import os
import subprocess
from os.path import join


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def delete_origin_file(dest_src: str, file_name: str):
    """
    删除某个文件，并返回他的路径，如果存在的话
    :param dest_src: 目标路径
    :param file_name: 文件名
    :return: 文件全路径
    """
    p = join(dest_src, file_name)
    if os.path.exists(p):
        os.remove(p)
    return p


def generate_configure(dest_src: str, file_name: str):
    """
    生成doxygen配置文件的指令
    :param dest_src: 目标目录
    :param file_name: configure 文件名
    :return: windows命令行指令
    """
    return 'cd {0} && doxygen -g {1} >/dev/null 2>&1'.format(dest_src, file_name)


def generate_run(dest_src: str, file_name: str):
    """
    生成doxygen run指令
    :param dest_src: 目标目录
    :param file_name: configure 文件名
    :return: windows命令行指令
    """
    return 'cd {0} && doxygen {1} >/dev/null 2>&1'.format(dest_src, file_name)


def edit_configure_file(source_src: str, dest_src: str, configure_path: str, project_name: str):
    """
    编辑doxygen的配置文件，输出xml文件
    :param source_src: 项目源路径
    :param dest_src: 输出目录
    :param configure_path: 配置文件路径
    :param project_name: 输出的项目名称
    :return:
    """
    # 需要修改如下字段
    # INPUT=source_src
    # OUTPUT_DIRECTORY=dest_src
    # GENERATE_HTML=NO
    # GENERATE_LATEX=NO
    # GENERATE_XML=YES
    # CALL_GRAPH=YES
    # CALLER_GRAPH=YES
    # EXTRACT_ALL=YES
    # EXTRACT_PRIVATE=YES
    # OPTIMIZE_OUTPUT_JAVA=YES
    file_data = ""
    with open(configure_path, "r", encoding="utf-8") as f:
        for line in f:
            new_line = line.replace(' ', '')
            if new_line.startswith('INPUT='):
                line = 'INPUT = ' + source_src + '\n'
            elif new_line.startswith('OUTPUT_DIRECTORY='):
                line = 'OUTPUT_DIRECTORY = ' + dest_src + '\n'
            elif new_line.startswith('GENERATE_HTML='):
                line = 'GENERATE_HTML = NO' + '\n'
            elif new_line.startswith('GENERATE_LATEX='):
                line = 'GENERATE_LATEX = NO' + '\n'
            elif new_line.startswith('GENERATE_XML='):
                line = 'GENERATE_XML = YES' + '\n'
            elif new_line.startswith('CALL_GRAPH='):
                line = 'CALL_GRAPH = YES' + '\n'
            elif new_line.startswith('EXTRACT_ALL='):
                line = 'EXTRACT_ALL = YES' + '\n'
            elif new_line.startswith('EXTRACT_PRIV_VIRTUAL='):
                line = 'EXTRACT_PRIV_VIRTUAL = YES' + '\n'
            elif new_line.startswith('EXTRACT_PRIVATE='):
                line = 'EXTRACT_PRIVATE = YES' + '\n'
            elif new_line.startswith('EXTRACT_STATIC='):
                line = 'EXTRACT_STATIC = YES' + '\n'
            elif new_line.startswith('OPTIMIZE_OUTPUT_JAVA='):
                line = 'OPTIMIZE_OUTPUT_JAVA = YES' + '\n'
            elif new_line.startswith('XML_OUTPUT='):
                line = 'XML_OUTPUT = ' + project_name + '\n'
            elif new_line.startswith('PROJECT_NAME='):
                line = 'PROJECT_NAME = ' + project_name + '\n'
            elif new_line.startswith('FILE_PATTERNS='):
                line = 'FILE_PATTERNS = *.c \\' + '\n'
            elif new_line.startswith('RECURSIVE='):
                line = 'RECURSIVE = YES' + '\n'
            elif new_line.startswith('CALLER_GRAPH='):
                line = 'CALLER_GRAPH = YES' + '\n'
            elif new_line.startswith('INPUT_FILE_ENCODING='):
                line = 'INPUT_FILE_ENCODING = UTF-8' + '\n'
            file_data += line
    with open(configure_path, "w", encoding="utf-8") as f:
        f.write(file_data)


def run_xml_doxygen(source_src: str, dest_src: str, project_name: str):
    """
    运行生成xml的doxygen程序，将会自动生成doxygen配置文件和分析文件
    :param source_src: 源项目路径
    :param dest_src: 输出目录
    :param project_name: 项目名称
    :return:
    """
    # 设置doxygen环境变量，它默认是root的环境变量
    os.environ['PATH'] = '/data1/shunliu/program/doxygen/bin:' + os.environ['PATH']
    file_name = project_name + '_Doxyfile'
    configure_path = delete_origin_file(dest_src, file_name)
    configure_command = generate_configure(dest_src, file_name)
    print(configure_command)
    subprocess.run(configure_command, shell=True)
    edit_configure_file(source_src, dest_src, configure_path, project_name)
    run_command = generate_run(dest_src, file_name)
    print(run_command)
    subprocess.run(run_command, shell=True)


def main_doxygen(repo_list: list[str]):
    print('start run doxygen program of repo list: {0}'.format(repo_list))
    for repo in repo_list:
        src_index = repo.find("/src")
        repo_index = repo.rfind("/", 0, src_index)
        project_name = repo[repo_index + 1: src_index]
        dest_src = join(repo[0: repo_index], 'doxygen')
        make_dir(dest_src)
        run_xml_doxygen(repo, dest_src, project_name)
        print('run {0} doxygen finished'.format(project_name))
    print("doxygen run finish")
