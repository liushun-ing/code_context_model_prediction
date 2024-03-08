"""
解析各个model的ast,并保存
ast 只解析一次，保存起来即可，太多了，很费时间
"""
import os
from os.path import join
import pandas as pd
import warnings
from tqdm.auto import tqdm

tqdm.pandas()
warnings.filterwarnings('ignore')

repo_root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


def get_parsed_source(input_file: str, output_file=None):
    """Parse code using pycparser

    it reads a Dataframe from `input_file` containing the node_id and
    code (input Java code) , applies the javalang to the code column and
    stores the resulting dataframe into `output_file`

    Args:
        input_file (str): Path to the input file
        output_file (str): Path to the output file

    """
    import javalang

    def parse_program(member):
        """
        call javalang to get ast tree for element

        :param member: (id, code)
        :return: ast tree
        """
        _id: str = member[0]
        tokens = javalang.tokenizer.tokenize(member[1])
        parser = javalang.parser.Parser(tokens)
        _id = _id[_id.find('_') + 1:]
        code_type = _id[:_id.find('_')]
        # 其实可以统一用 parse_member_declaration()
        if code_type == 'class' or code_type == 'interface':
            tree = parser.parse_class_or_interface_declaration()
        else:
            tree = parser.parse_member_declaration()
        return tree

    # 读java数据文件
    source = pd.read_csv(input_file, delimiter='\t')
    # print(source)
    source['tokens'] = source.progress_apply(parse_program, axis=1)
    source.columns = ['id', 'code']
    source.to_pickle(output_file)  # 保存解析的结果


def main_func(project_model_name: str):
    project_path = join(repo_root_path, project_model_name, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    # 读取code context model
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        java_code_path = join(model_path, 'processed_java_codes.tsv')
        # 如果不存在java_code，跳过处理
        if not os.path.exists(java_code_path):
            continue
        print('solving ast...')
        if os.path.exists(join(model_path, 'astnn_ast.pkl')):
            print('ast exist, now removed')
            os.remove(join(model_path, 'astnn_ast.pkl'))
        get_parsed_source(input_file=join(model_path, 'processed_java_codes.tsv'), output_file=join(model_path, 'astnn_ast.pkl'))


main_func('my_mylyn')
main_func('my_pde')
main_func('my_platform')
main_func('my_ecf')

