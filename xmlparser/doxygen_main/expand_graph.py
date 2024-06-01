"""
根据结构关系扩展 code context model 的相关 API
v_element.prot != 'package' 是为了剔除 static 代码块
"""
import random
from typing import Union

from xmlparser.doxmlparser.compound import DoxMemberKind, DoxCompoundKind
from xmlparser.doxygen_main import solve_graph
from xmlparser.doxygen_main.ClassEntity import ClassEntity
from xmlparser.doxygen_main.FieldEntity import FieldEntity
from xmlparser.doxygen_main.Graph import Graph, EdgeLabel
from xmlparser.doxygen_main.MethodEntity import MethodEntity
from xmlparser.doxygen_main.Metrics import RepoMetrics
from xmlparser.doxygen_main.Vertex import Vertex

function_like_kind = [DoxMemberKind.FUNCTION]
variable_like_kind = [DoxMemberKind.VARIABLE]


def get_label(element: Union[ClassEntity, MethodEntity, FieldEntity, None]):
    """
    根据不同类型，获取不同的元素的名称

    :param element: 元素
    :return: 作为 label 的名称字符串
    """
    if isinstance(element, ClassEntity):
        return element.compound_name.replace('::', '.')
    elif isinstance(element, MethodEntity):
        return element.full_name
    elif isinstance(element, FieldEntity):
        return element.qualified_name
    else:
        return ''


def select_random_percent(arr, percent):
    # 计算要挑选的元素数量
    num_to_select = int(len(arr) * percent)
    # 如果数量小于1，则设置为1
    num_to_select = max(num_to_select, 1)
    # 如果要挑选的元素数量大于数组长度，调整为数组长度
    num_to_select = min(num_to_select, len(arr))
    # 随机选择元素
    selected_indices = random.sample(range(len(arr)), num_to_select)
    # 获取选中的元素
    selected_elements = [arr[i] for i in selected_indices]
    return selected_elements


def expand_field(graph: Graph, repo_metrics: RepoMetrics, vertex: Vertex):
    """
    扩展字段：被声明和被调用，只关心节点

    :param graph: 图
    :param repo_metrics: doxygen矩阵
    :param vertex: 节点
    :return: 没有返回值
    """
    # 被声明
    v_ref_id = vertex.ref_id[:vertex.ref_id.rfind('_')]
    v_class = repo_metrics.get_class_by_id(v_ref_id)
    if v_class is not None:
        v_id = graph.get_vertex_id_by_ref_id(v_class.ref_id)
        # 如果该类不存在，则加入,并更新节点id
        if v_id == -1:
            if v_class.prot != 'package':
                graph.add_vertex(v_class.ref_id, v_class.kind, get_label(v_class))
    # 被调用
    # fie = repo_metrics.get_field_by_id(vertex.ref_id)
    # if fie is not None:
    #     selected_referenced = select_random_percent(fie.referenced_by, 0.5)
    #     for ref_by_id in selected_referenced:
    #         v_element = repo_metrics.get_element_by_id(ref_by_id)
    #         if v_element is not None:
    #             v_id = graph.get_vertex_id_by_ref_id(ref_by_id)
    #             if v_id == -1:  # 如果节点不存在，则加入,并更新节点id
    #                 if v_element.prot != 'package':
    #                     graph.add_vertex(ref_by_id, v_element.kind, get_label(v_element))


def expand_method(graph: Graph, repo_metrics: RepoMetrics, vertex: Vertex):
    """
    扩展方法：被声明，调用（主动和被动），实现（主动和被动），只关心节点

    :param graph: 图
    :param repo_metrics: doxygen矩阵
    :param vertex: 节点
    :return: 没有返回值
    """
    # 被声明
    v_ref_id = vertex.ref_id[:vertex.ref_id.rfind('_')]
    v_class = repo_metrics.get_class_by_id(v_ref_id)
    if v_class is not None:
        v_id = graph.get_vertex_id_by_ref_id(v_class.ref_id)
        # 如果该类不存在，则加入,并更新节点id
        if v_id == -1:
            if v_class.prot != 'package':
                graph.add_vertex(v_class.ref_id, v_class.kind, get_label(v_class))
    # 调用关系和实现关系
    met = repo_metrics.get_method_by_id(vertex.ref_id)
    if met is not None:
        # 主动调用
        selected_reference = []
        for ref_id in met.references:
            v_element = repo_metrics.get_element_by_id(ref_id)
            if v_element is not None:
                # 不要 field
                if not v_element.kind == 'variable':
                    selected_reference.append(ref_id)
        for ref_id in selected_reference:
            v_element = repo_metrics.get_element_by_id(ref_id)
            if v_element is not None:
                v_id = graph.get_vertex_id_by_ref_id(ref_id)
                if v_id == -1:
                    if v_element.prot != 'package':
                        graph.add_vertex(v_element.ref_id, v_element.kind, get_label(v_element))
        # 被调用
        for ref_by_id in met.referenced_by:
            v_element = repo_metrics.get_element_by_id(ref_by_id)
            if v_element is not None:
                v_id = graph.get_vertex_id_by_ref_id(ref_by_id)
                if v_id == -1:
                    if v_element.prot != 'package':
                        graph.add_vertex(v_element.ref_id, v_element.kind, get_label(v_element))
        # 主动实现
        for re_imp_id in met.reimplements:
            v_element = repo_metrics.get_element_by_id(re_imp_id)
            if v_element is not None:
                v_id = graph.get_vertex_id_by_ref_id(re_imp_id)
                if v_id == -1:
                    if v_element.prot != 'package':
                        graph.add_vertex(v_element.ref_id, v_element.kind, get_label(v_element))
        # 被实现
        for re_imp_by_id in met.reimplemented_by:
            v_element = repo_metrics.get_element_by_id(re_imp_by_id)
            if v_element is not None:
                v_id = graph.get_vertex_id_by_ref_id(re_imp_by_id)
                if v_id == -1:
                    if v_element.prot != 'package':
                        graph.add_vertex(v_element.ref_id, v_element.kind, get_label(v_element))


def expand_class(graph: Graph, repo_metrics: RepoMetrics, vertex: Vertex):
    """
    扩展类：声明，继承（类，接口），实现（类），被继承（类，接口），被实现（接口），这一步只是找到节点，并加入图中，后续在统一找边

    :param graph: 图
    :param repo_metrics: doxygen矩阵
    :param vertex: 节点
    :return: 没有返回值
    """
    cla = repo_metrics.get_class_by_id(vertex.ref_id)
    if cla is not None:
        # 声明的字段
        selected_fields = select_random_percent(cla.fields, 0.5)
        for c_f in selected_fields:
            v_id = graph.get_vertex_id_by_ref_id(c_f.ref_id)
            if v_id == -1:
                graph.add_vertex(c_f.ref_id, c_f.kind, c_f.qualified_name)
        # 找声明的方法
        selected_methods = select_random_percent(cla.methods, 0.5)
        for c_m in selected_methods:
            v_id = graph.get_vertex_id_by_ref_id(c_m.ref_id)
            if v_id == -1:
                if c_m.prot != 'package':
                    graph.add_vertex(c_m.ref_id, c_m.kind, c_m.full_name)
        # 找继承（类继承类，类实现接口，接口继承接口）
        for base_ref in cla.base_compound_ref:
            if base_ref != '':
                v_element = repo_metrics.get_class_by_id(base_ref)
                if v_element is not None:
                    v_id, v_kind = graph.get_vertex_id_and_kind_by_ref_id(base_ref)
                    if v_id == -1:
                        if v_element.prot != 'package':
                            graph.add_vertex(v_element.ref_id, v_element.kind, get_label(v_element))
        # 找被继承（类被类继承，接口被类实现，接口被接口继承）
        for derived_ref in cla.derived_compound_ref:
            if derived_ref != '':
                v_element = repo_metrics.get_class_by_id(derived_ref)
                if v_element is not None:
                    v_id, v_kind = graph.get_vertex_id_and_kind_by_ref_id(derived_ref)
                    if v_id == -1:
                        if v_element.prot != 'package':
                            graph.add_vertex(v_element.ref_id, v_element.kind, get_label(v_element))


def expand_graph_1_step(graph: Graph, repo_metrics: RepoMetrics, start: int):
    """
    一步扩展图，只考虑扩展的节点加入图中，然后调用其他函数完成边的加入

    :param graph: 图
    :param repo_metrics: doxygen矩阵
    :param start: 需要从哪个节点开始扩展，节点下标，也就是id
    :return: 扩展之前的节点数
    """
    origin_vertex_number = len(graph.vertices)
    for vertex in graph.vertices[start:]:
        if vertex.kind in variable_like_kind:
            expand_field(graph, repo_metrics, vertex)
        elif vertex.kind in function_like_kind:
            expand_method(graph, repo_metrics, vertex)
        elif vertex.kind == DoxCompoundKind.CLASS or vertex == DoxCompoundKind.INTERFACE:
            expand_class(graph, repo_metrics, vertex)
    return origin_vertex_number


def complete_graph(graph: Graph, repo_metrics: RepoMetrics):
    """
    完善扩展图后，补充节点之间的边关系

    :param graph: 图
    :param repo_metrics: doxygen矩阵
    :return: 无
    """
    for vertex in graph.vertices:
        if vertex.origin == 0:
            solve_graph.complete_edge_of_vertex(graph, repo_metrics, vertex)


def expand_model_graph(graph: Graph, repo_metrics: RepoMetrics, step: int):
    """
    根据步长step,扩展单个图

    :param graph: 图
    :param repo_metrics: 该图对应的doxygen矩阵
    :param step: 步长
    :return: 无
    """
    if step == 1:
        expand_graph_1_step(graph, repo_metrics, 0)
    elif step == 2:
        origin_number = expand_graph_1_step(graph, repo_metrics, 0)
        # 从origin_number后，再一次调用即可
        expand_graph_1_step(graph, repo_metrics, origin_number)
    elif step == 3:
        origin_number_1 = expand_graph_1_step(graph, repo_metrics, 0)
        # 从origin_number后，再两次调用即可
        origin_number_2 = expand_graph_1_step(graph, repo_metrics, origin_number_1)
        expand_graph_1_step(graph, repo_metrics, origin_number_2)
    complete_graph(graph, repo_metrics)


def add_location_to_field_and_method(graph: Graph, repo_metrics: RepoMetrics):
    """
    为 Method 和 Field 类型节点添加 location 信息

    :param graph: 图
    :param repo_metrics: doxygen矩阵
    :return: 无
    """
    for vertex in graph.vertices:
        if vertex.kind in function_like_kind:
            met = repo_metrics.get_method_by_id(vertex.ref_id)
            vertex.set_location(met.location)
        elif vertex.kind in variable_like_kind:
            fie = repo_metrics.get_field_by_id(vertex.ref_id)
            vertex.set_location(fie.location)


def expand_model(graph_list: list[Graph], all_repo_metrics: list[RepoMetrics], step: int):
    """
    根据step,扩展模型

    :param graph_list: 模型的图集合
    :param all_repo_metrics: 所有的doxygen矩阵
    :param step: 扩展步长
    :return: 无，因为就是在graph对象上做修改
    """
    for graph in graph_list:
        for repo_metrics in all_repo_metrics:
            if graph.repo_name == repo_metrics.repo_name:
                expand_model_graph(graph, repo_metrics, step)
                add_location_to_field_and_method(graph, repo_metrics)
    print("{} step expand model over~~~~~~~~~~~~~~~~".format(step))
