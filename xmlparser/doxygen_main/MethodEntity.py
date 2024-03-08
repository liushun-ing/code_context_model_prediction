from xmlparser.doxygen_main.LocationEntity import LocationEntity


class MethodEntity:
    """
    方法实体
    """
    ref_id: str  # id 唯一标识
    kind: str  # 分类类型 variable or function
    prot: str  # public private
    static: str  # 是否为静态 yes no
    return_type: str  # 返回值类型
    definition: str  # 定义字符串 IPresenceService[] org.eclipse.ecf.internal.presence.ui.Activator.getPresenceServices
    args_string: str  # 参数字符串 (xxx xxx, xxx xx)
    name: str  # 名字
    qualified_name: str  # 全名称
    full_name: str  # 包含参数的名称,用于后续比较code elements
    params: list[dict]  # 参数  包含 param_type 类型 和 declname 参数名
    location_file: str  # 所属文件路径字符串
    references: list[str]  # 引用的元素的id数组
    referenced_by: list[str]  # 被引用的元素的id数组
    reimplements: list[str]  # 实现的方法
    reimplemented_by: list[str]  # 被实现的方法
    location: LocationEntity

    def __init__(self):
        self.ref_id = ''
        self.kind = ''
        self.prot = ''
        self.static = ''
        self.return_type = ''
        self.full_name = ''
        self.definition = ''
        self.args_string = ''
        self.name = ''
        self.qualified_name = ''
        self.params = []
        self.location_file = ''
        self.references = []
        self.referenced_by = []
        self.reimplements = []
        self.reimplemented_by = []
        self.location = LocationEntity()

    def print(self):
        print("ref_id: {0}, qualified_name: {1}, definition: {2}, args_string: {3}, full_name: {4}, references: {5}, "
              "referencedBy: {6}, reimplements: {7}, reimplemented_by: {8}, location: {9}".format(
            self.ref_id, self.qualified_name, self.definition, self.args_string, self.full_name, self.references,
            self.referenced_by, self.reimplements, self.reimplemented_by, self.location
        ))

    def set_method_info(self, ref_id='', kind='', prot='', static='', return_type='', definition='', args_string='',
                        name='', full_name='',
                        qualified_name='', location_file=''):
        if not ref_id == '':
            self.ref_id = ref_id
        if not kind == '':
            self.kind = kind
        if not prot == '':
            self.prot = prot
        if not static == '':
            self.static = static
        if not return_type == '':
            self.return_type = return_type
        if not definition == '':
            self.definition = definition
        if not args_string == '':
            self.args_string = args_string
        if not name == '':
            self.name = name
        if not full_name == '':
            self.full_name = full_name
        if not qualified_name == '':
            self.qualified_name = qualified_name
        if not location_file == '':
            self.location_file = location_file

    def add_param(self, param: dict):
        self.params.append(param)

    def add_reference(self, reference: str):
        self.references.append(reference)

    def add_referenced_by(self, referenced_by: str):
        self.referenced_by.append(referenced_by)

    def add_reimplement(self, reimplement: str):
        self.reimplements.append(reimplement)

    def add_reimplemented_by(self, reimplemented_by: str):
        self.reimplemented_by.append(reimplemented_by)

    def set_location(self, location: LocationEntity):
        self.location = location
