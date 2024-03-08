class Edge:
    """
    图节点
    """
    start: int
    end: int  # class interface...
    label: str  # 节点element的全名
    origin: int  # origin=1表示是否是context model中的，而不是扩展的

    def __init__(self, start: int, end: int, label: str, origin: int):
        self.start = start
        self.end = end
        self.label = label
        self.origin = origin
