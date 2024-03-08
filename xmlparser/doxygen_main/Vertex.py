from xmlparser.doxygen_main.LocationEntity import LocationEntity


class Vertex:
    """
    图节点
    """
    id: int
    ref_id: str
    kind: str  # class interface...
    label: str  # 节点element的全名
    origin: int  # origin=1表示是否是context model中的，而不是扩展的
    location: LocationEntity

    def __init__(self, _id: int, ref_id: str, kind: str, label: str, origin: int, location=LocationEntity()):
        self.id = _id
        self.ref_id = ref_id
        self.kind = kind
        self.label = label
        self.origin = origin
        self.location = location

    def set_location(self, location=LocationEntity()):
        self.location = location
