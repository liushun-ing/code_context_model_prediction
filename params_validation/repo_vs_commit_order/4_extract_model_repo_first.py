"""
读取每个working periods，回退时间戳版本
查找是否存在对应元素，
将这部分文件提取出来（*这里一开始为了节省开销，只保留元素所在的文件，但是后续需要扩展，所以需要保留整个project*），运行doxygen,
保存输出的doxygen文件
然后分析doxygen输出文件，找出各个节点之间的关系，并生成上下文模型，保存到xml文件中

需要进行验证的：
commit number 1,2,3
commit和repo顺序：repo优先:每个repo一次探查多个commit, commit优先：还是一次探查一个，不断循环repo
"""
# 先用第一个测试一下
import shutil

from git import Repo
import os
import xml.etree.ElementTree as ET
from os.path import join

from params_validation.utils import solve_file, run_doxygen, copy_file
from xmlparser.doxygen_main import get_relations


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    return directory


root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')


def main_func(git_path: list[str], project_name: str, project_input: str):
    commit_index_threshold = 3  # 最近的commit_index的阈值 待取值为1，2，3
    file_path = join(os.path.dirname(os.path.realpath(__file__)), 'IQR_code_timestamp')
    file_dir_list = os.listdir(file_path)
    # 读取working periods
    for index_dir in file_dir_list:
        # 进入06文件夹
        if index_dir not in ['05']:
            continue
        print('---------------', index_dir)
        index_path = join(file_path, index_dir)
        project_list = os.listdir(index_path)
        # 进入项目目录
        for project_dir in project_list:
            if project_dir not in [project_input]:
                continue
            project_path = join(index_path, project_dir)
            xml_list = os.listdir(project_path)
            xml_list = [x[:x.find('.')] for x in xml_list]
            xml_list = sorted(xml_list, key=lambda x: int(x))  # 使用key=len在ubuntu不行，两个系统文件排序方法不一样
            xml_list = [x + ".xml" for x in xml_list]
            for xml_file in xml_list:
                # 这个编码都是出在 org.eclipse.mylyn.bugzilla.core/_bugzilla_core_plugin_8java.xml
                # 存在编码报错的情况，捕获错误后，自动跳过，后面filter out即可
                print("************** handle {}'s {} working period".format(project_dir, xml_file))
                # model_exist = join(root_path, f'my_{project_name}', f'repo_first_{commit_index_threshold}', xml_file[0:xml_file.find('.')], 'code_context_model.xml')
                # if os.path.exists(model_exist):
                #     print('exist....')
                #     continue
                # 创建目录，保存文件，方便运行doxygen
                make_root_path = make_dir(join(root_path, f'my_{project_name}', f'repo_first_{commit_index_threshold}',
                                               xml_file[0:xml_file.find('.')]))
                tree = ET.parse(join(project_path, xml_file))  # 拿到xml树
                # 获取XML文档的根元素
                root = tree.getroot()
                timestamps = root.find("timestamps")
                first_time = timestamps.find('first').text
                last_time = timestamps.find("last").text
                print('time: {0}'.format(last_time))
                code_element = root.find('code_elements')
                elements = code_element.findall('element')
                elements_text = []
                for element in elements:
                    elements_text.append(element.text)
                recent_commit = ''
                git_repository = ''
                # 所有项目git仓库都需要判断
                for git_repo in git_path:
                    print('git_repo ', git_repo)
                    git_repository = git_repo
                    repo = Repo(git_repo)
                    # 设置UTC时区
                    commits = list(repo.iter_commits(until=first_time + 'Z'))  # 找第一个之前的commit
                    if len(commits) == 0:
                        print('-------------no commits-----------')
                        continue
                    new_commit_index_threshold = commit_index_threshold
                    if len(commits) < commit_index_threshold:  # 需要判断边界
                        new_commit_index_threshold = len(commits)
                    for commit_index in range(new_commit_index_threshold):
                        print("handling recent {} commit".format(commit_index))
                        recent_commit = commits[commit_index]
                        newest_commit = list(repo.iter_commits(max_count=3))[0]  # 保存最新的那个提交
                        print("recent {0}:{1}".format(recent_commit, recent_commit.committed_datetime))
                        print("newest {0}:{1}".format(newest_commit, newest_commit.committed_datetime))
                        repo.git.reset(recent_commit, hard=True)
                        print("git reset to recent commit {0}".format(recent_commit))
                        try:
                            repo_list = solve_file.solve_element_directory(git_repo, elements_text)
                            if len(repo_list) > 0:
                                print("founded in {0} !!!!!!!!!!!!!!!!!!!!!!!!".format(git_repo))
                                # 只有找到了repo才进行下一步的分析,并且其他的不需要在分析了
                                copy_repo_list = copy_file.copy_element_file(make_root_path, repo_list, elements_text)
                                # print(copy_repo_list)
                                run_doxygen.main_doxygen(copy_repo_list)
                                # 分析完后，将版本重置到最新提交
                                repo.git.reset(newest_commit, hard=True)
                                print("git reset to newest commit {0}".format(newest_commit))
                                break
                            # 分析完后，将版本重置到最新提交
                            repo.git.reset(newest_commit, hard=True)
                            print("git reset to newest commit {0}".format(newest_commit))
                        except Exception as e:
                            print('Error Error Error >>>>>>------', e)
                            repo.git.reset(newest_commit, hard=True)
                            print("git reset to newest commit {0}".format(newest_commit))
                    else:
                        print("3 times commits finding completed!!!")
                        continue  # 如果三次都没找到，就进入下一个repo的查找
                    break  # 如果某一个找到了，就直接跳过外层的循环
                # 解析关系，生成图
                try:
                    graph_list, valid_list = get_relations.main_func(make_root_path, elements_text)
                    if len(graph_list) > 0:
                        # 更新code elements状态
                        for element in elements:
                            element.set('valid', str(valid_list[elements.index(element)]))
                        # 写图文件,将几个图组合在一起，就是代码上下文模型
                        model_path = join(make_root_path, 'code_context_model.xml')
                        model_root = ET.Element("code_context_model")
                        model_root.set('commit', str(recent_commit))
                        model_root.set('git_repository', str(git_repository))
                        model_root.set('total', str(len(graph_list)))
                        model_root.set('first_time', first_time)
                        model_root.set('last_time', last_time)
                        for graph in graph_list:
                            graph_node = ET.SubElement(model_root, 'graph')
                            graph_node.set('repo_name', graph.repo_name)
                            graph_node.set('repo_path', graph.repo_path)
                            vertices = ET.SubElement(graph_node, 'vertices')
                            vertices.set('total', str(len(graph.vertices)))
                            for vertex in graph.vertices:
                                v_node = ET.SubElement(vertices, 'vertex')
                                v_node.set('id', str(vertex.id))
                                v_node.set('ref_id', vertex.ref_id)
                                v_node.set('kind', vertex.kind)
                                v_node.set('label', vertex.label)
                            edges = ET.SubElement(graph_node, 'edges')
                            edges.set('total', str(len(graph.edges)))
                            for edge in graph.edges:
                                e_node = ET.SubElement(edges, 'edge')
                                e_node.set('start', str(edge.start))
                                e_node.set('end', str(edge.end))
                                e_node.set('label', edge.label)
                        model_tree = ET.ElementTree(model_root)
                        model_tree.write(model_path)
                    else:
                        for element in elements:
                            element.set('valid', '0')
                    tree.write(join(project_path, xml_file))
                    print('~~~~~~one working period finished~~~~~~')
                except Exception as e:
                    print('extractor context model error!!!!!!!!!!', e)
                    continue


# ecf
# main_func([join(root_path, 'ecf', 'ecf')], 'ecf', 'ECF')
# pde
# main_func([join(root_path, 'pde',  'eclipse.pde')], 'pde', 'PDE') #101
# platform
# main_func([
#     join(root_path, 'platform', 'eclipse.platform'),
#     join(root_path, 'platform', 'eclipse.platform.swt'),
#     join(root_path, 'platform', 'eclipse.platform.ui'),
#     join(root_path, 'platform', 'eclipse.platform.releng.buildtools'),
# ], 'platform', 'Platform')
# # mylyn
# print(root_path)
main_func([
    join(root_path, 'mylyn', 'org.eclipse.mylyn'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.context.mft'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.incubator'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.builds'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.commons'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.context'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.docs'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.reviews'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.tasks'),
    join(root_path, 'mylyn', 'org.eclipse.mylyn.versions')
], 'mylyn', 'Mylyn')
