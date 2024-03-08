from xmlparser.doxygen_main.Edge import Edge
from xmlparser.doxygen_main.Vertex import Vertex


class EdgeLabel:
    DECLARE = 'declares'
    CALL = 'calls'
    INHERIT = 'inherits'
    IMPLEMENT = 'implements'


class Graph:
    """
    关系图
    """
    repo_name: str  # repo_name
    repo_path: str  # repo_path
    vertices: list[Vertex]  # { id, ref_id, kind, label, origin } origin=1表示是否是context model中的，而不是扩展的
    edges: list[Edge]  # { start, end, label, origin }

    def __init__(self):
        self.repo_name = ''
        self.repo_path = ''
        self.vertices = []
        self.edges = []

    def print(self):
        print("repo_name: {0}, vertices: {0}, edges: {1}".format(self.repo_name, self.vertices, self.edges))

    def set_repo_name(self, repo_name: str):
        self.repo_name = repo_name

    def set_repo_path(self, repo_path: str):
        self.repo_path = repo_path

    def add_whole_vertex(self, vertex: Vertex):
        vertex.id = len(self.vertices)
        self.vertices.append(vertex)

    def add_vertex(self, ref_id: str, kind: str, label: str):
        self.vertices.append(Vertex(
            _id=len(self.vertices),
            ref_id=ref_id,
            kind=kind,
            label=label,
            origin=0
        ))
        return len(self.vertices)

    def add_vertex_origin(self, ref_id: str, kind: str, label: str):
        self.vertices.append(Vertex(
            _id=len(self.vertices),
            ref_id=ref_id,
            kind=kind,
            label=label,
            origin=1
        ))
        return len(self.vertices)

    def add_whole_edge(self, edge: Edge):
        self.edges.append(edge)

    def add_edge(self, start: int, end: int, label: str):
        if start < 0 or start >= len(self.vertices) or end < 0 or end >= len(self.vertices) or label == '':
            return
        for edge in self.edges:
            if edge.start == start and edge.end == end:
                return
        self.edges.append(Edge(
            start=start,
            end=end,
            label=label,
            origin=0
        ))

    def add_edge_origin(self, start: int, end: int, label: str):
        if start < 0 or start >= len(self.vertices) or end < 0 or end >= len(self.vertices) or label == '':
            return
        for edge in self.edges:
            if edge.start == start and edge.end == end:
                return
        self.edges.append(Edge(
            start=start,
            end=end,
            label=label,
            origin=1
        ))

    def get_vertex_by_id(self, _id: int):
        """
        根据id获取vertex

        :param _id: id
        :return: vertex，否则为 None
        """
        for vertex in self.vertices:
            if vertex.id == _id:
                return vertex
        return None

    def get_vertex_id_by_ref_id(self, ref_id: str):
        """
        根据ref_id查找是否存在匹配的顶点

        :param ref_id: ref_id
        :return: 顶点的 id，没有则返回空字符串
        """
        for vertex in self.vertices:
            if vertex.ref_id == ref_id:
                return vertex.id
        return -1

    def get_vertex_id_and_kind_by_ref_id(self, ref_id: str):
        """
        根据ref_id查找是否存在匹配的顶点

        :param ref_id: ref_id
        :return: 顶点的 id，没有则返回空字符串
        """
        for vertex in self.vertices:
            if vertex.ref_id == ref_id:
                return vertex.id, vertex.kind
        return -1, ''
