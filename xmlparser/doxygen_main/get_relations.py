import os
from os.path import join, isdir

from xmlparser import doxmlparser
from xmlparser.doxmlparser.compound import DoxCompoundKind, DoxMemberKind, MixedContainer
from xmlparser.doxygen_main import get_standard_elements, solve_graph
from xmlparser.doxygen_main.ClassEntity import ClassEntity
from xmlparser.doxygen_main.FieldEntity import FieldEntity
from xmlparser.doxygen_main.Graph import Graph
from xmlparser.doxygen_main.LocationEntity import LocationEntity
from xmlparser.doxygen_main.MethodEntity import MethodEntity
from xmlparser.doxygen_main.Metrics import RepoMetrics


def linked_text_to_string(linked_text):
    res_str = ''
    if linked_text:
        for text_or_ref in linked_text.content_:
            if text_or_ref.getCategory() == MixedContainer.CategoryText:
                res_str += text_or_ref.getValue()
            else:
                res_str += text_or_ref.getValue().get_valueOf_()
    return res_str


def parse_members(section_def, class_entity: ClassEntity):
    function_like_kind = [DoxMemberKind.FUNCTION]
    variable_like_kind = [DoxMemberKind.VARIABLE]
    for member_def in section_def.get_memberdef():
        # print("processing member_der {}".format(member_def.get_qualifiedname()))
        # method
        if member_def.get_kind() in function_like_kind:
            method_entity = MethodEntity()
            method_entity.set_method_info(
                ref_id=member_def.get_id(),
                prot=member_def.get_prot(),
                static=member_def.get_static(),
                kind=member_def.get_kind(),
                return_type=linked_text_to_string(member_def.get_type()),
                definition=member_def.get_definition(),
                name=member_def.get_name(),
                qualified_name=member_def.get_qualifiedname(),
                args_string=member_def.get_argsstring(),
                location_file=member_def.get_location().get_file()
            )
            param_str = []
            for param in member_def.get_param():
                name = param.get_declname()
                param_type = linked_text_to_string(param.get_type())
                # if param.get_type().get_ref():
                #     param_type = param.get_type().get_ref().get_refid()
                # else:
                #     param_type = param.get_type().valueOf_
                method_entity.add_param({
                    'name': name,
                    'param_type': param_type.split(' ')[-1]
                })
                param_str.append(param_type)
            method_entity.set_method_info(
                full_name=member_def.get_qualifiedname() + "(" + ','.join(param_str) + ")"
            )
            # 设置location
            method_entity.set_location(LocationEntity(
                file=member_def.get_location().get_file(),
                line=member_def.get_location().get_line(),
                column=member_def.get_location().get_column(),
                body_file=member_def.get_location().get_bodyfile(),
                body_start=member_def.get_location().get_bodystart(),
                body_end=member_def.get_location().get_bodyend()
            ))
            # 添加调用和被调用
            for reference in member_def.get_references():
                method_entity.add_reference(reference.get_refid())
            for referenced_by in member_def.get_referencedby():
                method_entity.add_referenced_by(referenced_by.get_refid())
            # 添加实现和被实现
            for reimplement in member_def.get_reimplements():
                method_entity.add_reimplement(reimplement.get_refid())
            for reimplemented_by in member_def.get_reimplementedby():
                method_entity.add_reimplemented_by(reimplemented_by.get_refid())
            class_entity.add_method(method_entity)
        # field
        elif member_def.get_kind() in variable_like_kind:
            field_entity = FieldEntity()
            field_entity.set_field_info(
                ref_id=member_def.get_id(),
                prot=member_def.get_prot(),
                static=member_def.get_static(),
                kind=member_def.get_kind(),
                field_type=linked_text_to_string(member_def.get_type()),
                definition=member_def.get_definition(),
                name=member_def.get_name(),
                qualified_name=member_def.get_qualifiedname(),
                initializer=linked_text_to_string(member_def.get_initializer()),
                location_file=member_def.get_location().get_file()
            )
            # 设置location
            field_entity.set_location(LocationEntity(
                file=member_def.get_location().get_file(),
                line=member_def.get_location().get_line(),
                column=member_def.get_location().get_column(),
                body_file=member_def.get_location().get_bodyfile(),
                body_start=member_def.get_location().get_bodystart(),
                body_end=member_def.get_location().get_bodyend()
            ))
            for referenced_by in member_def.get_referencedby():
                field_entity.add_referenced_by(referenced_by.get_refid())
            class_entity.add_field(field_entity)


def parse_sections(compound_def, class_entity: ClassEntity):
    for section_def in compound_def.get_sectiondef():
        # print("processing section_def {} kind".format(section_def.get_kind()))
        parse_members(section_def, class_entity)


def parse_compound(repo_path: str, id: str, metrics: RepoMetrics):
    root_obj = doxmlparser.compound.parse(repo_path + "/" + id + ".xml", True)
    for compound_def in root_obj.get_compounddef():
        kind = compound_def.get_kind()
        class_entity: ClassEntity = None
        if kind == DoxCompoundKind.CLASS or kind == DoxCompoundKind.INTERFACE:
            class_entity = ClassEntity()
            class_entity.set_class_info(
                ref_id=compound_def.get_id(),
                kind=compound_def.get_kind(),
                prot=compound_def.get_prot(),
                compound_name=compound_def.get_compoundname(),
            )
            for base_ref in compound_def.get_basecompoundref():
                ref_id = base_ref.get_refid() if base_ref.get_refid() else base_ref.valueOf_
                class_entity.add_base_compound_ref(ref_id)
            for derived_ref in compound_def.get_derivedcompoundref():
                ref_id = derived_ref.get_refid() if derived_ref.get_refid() else derived_ref.valueOf_
                class_entity.add_derived_compound_ref(ref_id)
            for inner_class in compound_def.get_innerclass():
                ref_id = inner_class.get_refid() if inner_class.get_refid() else inner_class.valueOf_
                class_entity.add_inner_class(ref_id)
        else:
            continue
        parse_sections(compound_def, class_entity)
        metrics.add_class_entity(class_entity)


def parse_index(repo_path: str, repo: str):
    metrics = RepoMetrics()
    metrics.set_repo_info(repo, repo_path)
    root_obj = doxmlparser.index.parse(repo_path + "/index.xml", True)
    for compound in root_obj.get_compound():  # for each compound defined in the index
        # print("Processing {0}...".format(compound.get_name()))
        parse_compound(repo_path, compound.get_refid(), metrics)
    # print("{0} parse finished".format(repo_path))
    return metrics


def parse_working_periods(period_path: str):
    """
    解析working periods的doxygen文件

    :param period_path: 路径
    :return: 返回解析的项目metrics数组
    """
    period_path = join(period_path, 'doxygen')
    all_repo_metrics: list[RepoMetrics] = []
    repo_dir_list = os.listdir(period_path)
    for repo in repo_dir_list:
        repo_path = join(period_path, repo)
        print(repo_path)
        if not isdir(repo_path):
            continue
        all_repo_metrics.append(parse_index(repo_path, repo))
    return all_repo_metrics


def solve_doxygen_metrics(repo_path: str):
    """
    根据项目目录，解析每个项目的 doxygen 矩阵

    :param repo_path: 每个working period的项目根目录
    :return: doxygen矩阵列表
    """
    dir_list = os.listdir(repo_path)  # 如果该路径下没有文件夹，说明没有从git上找到匹配的commit
    if len(dir_list) == 0:
        return []
    all_repo_metrics = parse_working_periods(repo_path)  # 分项目解析repo_metrics
    # for m in all_repo_metrics:
    #     m.print()
    return all_repo_metrics


def main_func(repo_path: str, code_elements: list[str]):
    """
    运行doxygen之后，根据IQR_code_elements中的element对doxygen的输出文件解析，并生成code context model

    :param repo_path: 该working period项目文件存放的的目录
    :param code_elements: 解析出来的所有elements
    :return: 解析出来的图集合，以及每个elements的匹配结果集合，1找到了0未找到
    """
    graph_list: list[Graph] = []
    valid_list: list[int] = []
    all_repo_metrics = solve_doxygen_metrics(repo_path)
    if len(all_repo_metrics) == 0:
        return graph_list, valid_list
    standard_elements = get_standard_elements.solve(code_elements)
    print(standard_elements)
    graph_list, valid_list = solve_graph.solve(all_repo_metrics, standard_elements)
    print(graph_list, valid_list)
    return graph_list, valid_list
