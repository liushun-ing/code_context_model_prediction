"""
根据metrics和standard elements解析关系的工具包
"""
from xmlparser.doxmlparser.compound import DoxCompoundKind, DoxMemberKind
from xmlparser.doxygen_main.Graph import Graph, EdgeLabel
from xmlparser.doxygen_main.Metrics import RepoMetrics
from xmlparser.doxygen_main.Vertex import Vertex

function_like_kind = [DoxMemberKind.FUNCTION]
variable_like_kind = [DoxMemberKind.VARIABLE]


def element_exist(all_repo_metrics: list[RepoMetrics], element: list[str]):
    """
    判断elements是否在metrics中存在，如果存在返回匹配的ref_id和kind

    :param all_repo_metrics: 所有的doxygen矩阵
    :param element: 代码元素字符串
    :return: ref_id, kind 如果有的话
    """
    for metric in all_repo_metrics:
        if metric.repo_name != element[0]:
            continue
        for c in metric.classes:
            c_name = c.compound_name.replace('::', '.')
            if c_name == element[1]:
                return c.ref_id, c.kind
            for f in c.fields:
                if f.qualified_name == element[1]:
                    return f.ref_id, f.kind
            for m in c.methods:
                if m.full_name == element[1]:
                    return m.ref_id, m.kind
    return '', ''


def transfer_elements(all_repo_metrics: list[RepoMetrics], standard_elements: list[list[str]]):
    """
    转换code elements格式，将其转换为dict格式，{ref_id:xxx, name:xxx}

    :param all_repo_metrics: doxygen矩阵
    :param standard_elements: 标准的code elements字符串
    :return: 返回dict，{repo: [elements], repo: [elements]}
    """
    new_elements = dict()
    valid_list = []
    ref_id_list = []
    for se in standard_elements:
        if se[0] == '' or se[1] == '':
            valid_list.append(0)
            continue
        ref_id, kind = element_exist(all_repo_metrics, se)
        if ref_id == '':
            valid_list.append(0)
            continue
        else:
            valid_list.append(1)
            if ref_id not in ref_id_list:  # 对结果去重
                ref_id_list.append(ref_id)
                if se[0] in new_elements.keys():
                    new_elements.get(se[0]).append({
                        'ref_id': ref_id,
                        'kind': kind,
                        'name': se[1]
                    })
                else:
                    new_elements[se[0]] = [{
                        'ref_id': ref_id,
                        'kind': kind,
                        'name': se[1]
                    }]
    return new_elements, valid_list


def complete_edge_of_vertex(graph: Graph, metric: RepoMetrics, vertex: Vertex):
    """
    给定节点，完善他与图中的其他节点的边

    :param graph: 图
    :param metric: doxygen矩阵
    :param vertex: 节点
    :return: 无
    """
    kind = vertex.kind
    ref_id = vertex.ref_id
    origin_id = vertex.id
    if kind == DoxCompoundKind.CLASS or kind == DoxCompoundKind.INTERFACE:
        # 如果是类，找继承关系和声明关系
        cla = metric.get_class_by_id(ref_id)
        if cla is None:
            return
        # 先找继承,目前只考虑直接继承
        for base_ref in cla.base_compound_ref:
            if base_ref != '':
                v_id, v_kind = graph.get_vertex_id_and_kind_by_ref_id(base_ref)
                if v_id != -1:
                    # 需要区分是接口还是类，关系到边是inherits 还是 implements
                    if v_kind == DoxCompoundKind.CLASS:  # 能继承类的只有类
                        graph.add_edge(
                            start=origin_id,
                            end=v_id,
                            label=EdgeLabel.INHERIT
                        )
                    elif v_kind == DoxCompoundKind.INTERFACE:  # 能继承接口的，可以是类实现接口，也可以是接口继承接口
                        if kind == DoxCompoundKind.INTERFACE:
                            graph.add_edge(
                                start=origin_id,
                                end=v_id,
                                label=EdgeLabel.INHERIT
                            )
                        else:
                            graph.add_edge(
                                start=origin_id,
                                end=v_id,
                                label=EdgeLabel.IMPLEMENT
                            )
        # 找被继承,或者被实现
        for derived_ref in cla.derived_compound_ref:
            if derived_ref != '':
                v_id, v_kind = graph.get_vertex_id_and_kind_by_ref_id(derived_ref)
                if v_id != -1:
                    if v_kind == DoxCompoundKind.CLASS:  # 能被类实现的，可以是类被类继承，也可以接口被类实现
                        if kind == DoxCompoundKind.CLASS:
                            graph.add_edge(
                                start=v_id,
                                end=origin_id,
                                label=EdgeLabel.INHERIT
                            )
                        else:
                            graph.add_edge(
                                start=v_id,
                                end=origin_id,
                                label=EdgeLabel.IMPLEMENT
                            )
                    elif v_kind == DoxCompoundKind.INTERFACE:  # 能被接口继承的只有接口
                        if kind == DoxCompoundKind.INTERFACE:
                            graph.add_edge(
                                start=v_id,
                                end=origin_id,
                                label=EdgeLabel.INHERIT
                            )
        # 找声明的字段
        for c_f in cla.fields:
            v_id = graph.get_vertex_id_by_ref_id(c_f.ref_id)
            if v_id != -1:
                graph.add_edge(
                    start=origin_id,
                    end=v_id,
                    label=EdgeLabel.DECLARE
                )
        # 找声明的方法
        for c_m in cla.methods:
            v_id = graph.get_vertex_id_by_ref_id(c_m.ref_id)
            if v_id != -1:
                graph.add_edge(
                    start=origin_id,
                    end=v_id,
                    label=EdgeLabel.DECLARE
                )
    elif kind in function_like_kind:
        # 被声明,由于该API在后续扩展图时要用到，所以声明需要加上，但是这里改动不影响前面
        v_ref_id = ref_id[:ref_id.rfind('_')]
        v_id = graph.get_vertex_id_by_ref_id(v_ref_id)
        # 如果该类不存在，则加入,并更新节点id
        if v_id != -1:
            graph.add_edge(start=v_id, end=origin_id, label=EdgeLabel.DECLARE)
        # 如果是方法，要找声明和调用，声明在类的时候会完成，调用要找主动和被动
        met = metric.get_method_by_id(ref_id)
        for ref_by_id in met.referenced_by:
            v_id = graph.get_vertex_id_by_ref_id(ref_by_id)
            if v_id != -1:
                # 注意是被调用关系
                graph.add_edge(
                    start=v_id,
                    end=origin_id,
                    label=EdgeLabel.CALL
                )
        for ref_id in met.references:
            v_id = graph.get_vertex_id_by_ref_id(ref_id)
            if v_id != -1:
                # 注意是调用关系
                graph.add_edge(
                    start=origin_id,
                    end=v_id,
                    label=EdgeLabel.CALL
                )
        # 方法还需要找实现关系
        for ref_by_id in met.reimplemented_by:
            v_id = graph.get_vertex_id_by_ref_id(ref_by_id)
            if v_id != -1:
                # 注意是被实现关系
                graph.add_edge(
                    start=v_id,
                    end=origin_id,
                    label=EdgeLabel.IMPLEMENT
                )
        for ref_by_id in met.reimplements:
            v_id = graph.get_vertex_id_by_ref_id(ref_by_id)
            if v_id != -1:
                # 注意是实现关系
                graph.add_edge(
                    start=origin_id,
                    end=v_id,
                    label=EdgeLabel.IMPLEMENT
                )
    elif kind in variable_like_kind:
        # 被声明,由于该API在后续扩展图时要用到，所以声明需要加上
        v_ref_id = ref_id[:ref_id.rfind('_')]
        v_id = graph.get_vertex_id_by_ref_id(v_ref_id)
        # 如果该类不存在，则加入,并更新节点id
        if v_id != -1:
            graph.add_edge(start=v_id, end=origin_id, label=EdgeLabel.DECLARE)
        # 如果是字段，需要找声明和调用，声明在类的时候会完成，只需要找被调用即可
        fie = metric.get_field_by_id(ref_id)
        for ref_by_id in fie.referenced_by:
            v_id = graph.get_vertex_id_by_ref_id(ref_by_id)
            if v_id != -1:
                # 注意是被调用关系
                graph.add_edge(
                    start=v_id,
                    end=origin_id,
                    label=EdgeLabel.CALL
                )


def solve_repo_relation(metric: RepoMetrics, dict_elements: list[dict[str, str]]):
    """
    解析单个repo的关系，返回一个图结果，包含节点集合，和边集合

    :param metric: doxygen矩阵
    :param dict_elements: 标准化的code elements
    :return: 关系图
    """
    graph = Graph()
    # 添加所有节点
    for ele in dict_elements:
        graph.add_vertex(
            ref_id=ele.get('ref_id'),
            kind=ele.get('kind'),
            label=ele.get('name')
        )
    for vertex in graph.vertices:
        complete_edge_of_vertex(graph, metric, vertex)
    return graph


def solve(all_repo_metrics: list[RepoMetrics], standard_elements: list[list[str]]):
    """
    根据repo矩阵，以及标准的element name,解析图关系

    :param all_repo_metrics: doxygen解析的repo metrics
    :param standard_elements: 标准形式的elements
    :return: 图集合（也就是代码上下文）和element valid状态集合
    """
    dict_elements, valid_list = transfer_elements(all_repo_metrics, standard_elements)
    # print(dict_elements, valid_list)
    graph_list: list[Graph] = []
    for repo in dict_elements.keys():
        for me in all_repo_metrics:
            if me.repo_name == repo:
                graph = solve_repo_relation(me, dict_elements.get(repo))
                graph.set_repo_name(me.repo_name)  # add set repo_name
                graph.set_repo_path(me.repo_location)  # add set repo_location
                graph_list.append(graph)
                # graph.print()
                break
    return graph_list, valid_list
