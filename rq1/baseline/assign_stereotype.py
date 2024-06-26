import os
from os.path import join

import xml.etree.ElementTree as ET

import pandas as pd

from dataset_split_util import get_models_by_ratio

root_path = join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'params_validation',
                 'git_repo_code')


def merge_stereotypes():
    data = []
    for s in ['0570', '1000', '2000', '2500', '3000', '3500', '4000']:
        with open(f"./stereotype/{s}.txt", "r") as f:
            lines = f.readlines()
            print(s, len(lines))
            for line in lines:
                line = line.replace('NULL', 'OTHER')
                if len(line.strip('\n').split(' ')) > 4:
                    print(line)
                data.append(line.strip('\n').split(' ', 4))
    stereotypes = pd.DataFrame(data, columns=['model', 'project', 'label', 'stereotype'])
    stereotypes.to_csv('./stereotype/all_types.tsv', sep=' ', index=False,
                       header=['model', 'project', 'label', 'stereotype'])
    print(len(data))


def main_func(project_model_name: str, step: int):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    stereotypes = pd.read_csv('./stereotype/all_types.tsv', sep=' ')
    print(stereotypes)
    # 读取code context model
    model_dir_list = get_models_by_ratio(project_model_name, 0.9, 1)
    # model_dir_list = os.listdir(project_path)
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    # index = model_dir_list.index('459')
    all_stereotypes = {'FIELD'}
    for model_dir in model_dir_list:
        # print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        # model_file = join(model_path, 'code_context_model.xml')
        model_file = join(model_path, f'new1_{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        sub_stereotypes = stereotypes[stereotypes['model'] == int(model_dir)]
        # if len(sub_stereotypes) == 0:
        #     sub_stereotypes = stereotypes
        # print(len(sub_stereotypes))
        graphs = code_context_model.findall("graph")
        for graph in graphs:
            repo = graph.get('repo_name')
            sub_sub_stereotypes = sub_stereotypes[sub_stereotypes['project'] == repo]
            # print(len(sub_sub_stereotypes))
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                kind, label = vertex.get('kind'), vertex.get('label')
                label = label.replace("final ", '').replace(' ', '').replace('...', '').replace(
                    '@SuppressWarnings("unchecked")', '')
                # print(label)
                if kind == 'variable':
                    vertex.set('stereotype', 'FIELD')
                else:
                    s = sub_sub_stereotypes[sub_sub_stereotypes['label'].str.contains(label, regex=False)]
                    if len(s) != 0:
                        # print(s['stereotype'])
                        vertex.set('stereotype', s['stereotype'].values[0])
                        all_stereotypes.add(s['stereotype'].values[0])
                    else:
                        if kind == 'function':
                            func = label[label.rfind('.') + 1:]
                            label = label[:label.rfind('.')]
                            cla = label[label.rfind('.') + 1:]
                            s = sub_sub_stereotypes[
                                sub_sub_stereotypes['label'].str.contains(f'{cla}.{func}', regex=False)]
                            if len(s) != 0:
                                # print(s['stereotype'].values[0])
                                vertex.set('stereotype', s['stereotype'].values[0])
                                all_stereotypes.add(s['stereotype'].values[0])
                            else:
                                label = label[:label.rfind('.')]
                                pack = label[label.rfind('.') + 1:]
                                s = sub_sub_stereotypes[
                                    sub_sub_stereotypes['label'].str.contains(f'{pack}.{cla}.{func}', regex=False)]
                                if len(s) != 0:
                                    # print(s['stereotype'].values[0])
                                    vertex.set('stereotype', s['stereotype'].values[0])
                                    all_stereotypes.add(s['stereotype'].values[0])
                                else:
                                    print(f'NOTFOUND-method-{vertex.get("label")}-{pack}.{cla}.{func}')
                                    vertex.set('stereotype', 'NOTFOUND')
                                    all_stereotypes.add('NOTFOUND')
                        else:
                            cla = label[label.rfind('.') + 1:]
                            s = sub_sub_stereotypes[sub_sub_stereotypes['label'].str.contains(cla, regex=False)]
                            if len(s) != 0:
                                # print(s['stereotype'].values[0])
                                vertex.set('stereotype', s['stereotype'].values[0])
                                all_stereotypes.add(s['stereotype'].values[0])
                            else:
                                label = label[:label.rfind('.')]
                                pack = label[label.rfind('.') + 1:]
                                s = sub_sub_stereotypes[
                                    sub_sub_stereotypes['label'].str.contains(f'{pack}.{cla}', regex=False)]
                                if len(s) != 0:
                                    # print(s['stereotype'].values[0])
                                    vertex.set('stereotype', s['stereotype'].values[0])
                                    all_stereotypes.add(s['stereotype'].values[0])
                                else:
                                    print(f'NOTFOUND-class-{vertex.get("label")}-{pack}.{cla}')
                                    vertex.set('stereotype', 'NOTFOUND')
                                    all_stereotypes.add('NOTFOUND')
        tree.write(join(model_path, f'new1_{step}_step_expanded_model.xml'))
        # tree.write(join(model_path, 'code_context_model.xml'))
        print('stereotype {} code context model over~~~~~~~~~~~~'.format(model_file))
    print(all_stereotypes, len(all_stereotypes))


if __name__ == '__main__':
    # merge_stereotypes()
    # main_func('my_mylyn', step=1)
    # main_func('my_mylyn', step=2)
    main_func('my_mylyn', step=1)
