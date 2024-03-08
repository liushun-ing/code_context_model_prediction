from xmlparser.doxygen_main.LocationEntity import LocationEntity


class FieldEntity:
    """
    字段实体
    """
    ref_id: str  # id 唯一标识
    kind: str  # 分类类型 variable or function
    prot: str  # public or private
    static: str  # 是否为静态 yes no
    field_type: str  # 字段类型
    definition: str  # 定义字符串
    name: str  # 名字
    qualified_name: str  # 全名称  xxx.xxx.xxx.xx
    initializer: str  # 初始字符串 =xxx
    location_file: str  # 位置文件路径字符串
    referenced_by: list[str]  # 被引用的id
    location: LocationEntity

    def __init__(self):
        self.ref_id = ''
        self.kind = ''
        self.prot = ''
        self.static = ''
        self.field_type = ''
        self.definition = ''
        self.name = ''
        self.qualified_name = ''
        self.initializer = ''
        self.location_file = ''
        self.referenced_by = []
        self.location = LocationEntity()

    def print(self):
        print("name: {0}, kind: {1}, prot: {2}, static: {3}, field_type: {4}, definition: {5}, qualified_name: {6}, "
              "initializer: {7}, location_file: {8}, ref_id: {9}, referencedBy: {10}".format(
            self.name, self.kind, self.prot, self.static, self.field_type, self.definition,
            self.qualified_name, self.initializer, self.location_file, self.ref_id, self.referenced_by)
        )

    def set_field_info(self, ref_id='', kind='', prot='', static='', field_type='', definition='', name='',
                       qualified_name='', location_file='', initializer=''):
        if not ref_id == '':
            self.ref_id = ref_id
        if not kind == '':
            self.kind = kind
        if not prot == '':
            self.prot = prot
        if not static == '':
            self.static = static
        if not field_type == '':
            self.field_type = field_type
        if not definition == '':
            self.definition = definition
        if not initializer == '':
            self.initializer = initializer
        if not name == '':
            self.name = name
        if not qualified_name == '':
            self.qualified_name = qualified_name
        if not location_file == '':
            self.location_file = location_file

    def add_referenced_by(self, referenced_by: str):
        self.referenced_by.append(referenced_by)

    def set_location(self, location: LocationEntity):
        self.location = location
