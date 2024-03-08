from xmlparser.doxygen_main.FieldEntity import FieldEntity
from xmlparser.doxygen_main.MethodEntity import MethodEntity


class ClassEntity:
    """
    解析类/接口的定义，两者的区别在于kind属性=class or interface
    """
    ref_id: str  # id 唯一标识
    kind: str  # class/interface
    prot: str  # public/private
    compound_name: str  # 名字 org::eclipse::ecf::presence::ui::handlers::BrowseDialog::RosterItemsComparator
    base_compound_ref: list[str]  # 继承的类数组 如果有id 则为id 否则为名字
    derived_compound_ref: list[str]  # 被继承的类数组 如果有id 则为id 否则为名字
    inner_class: list[str]  # 内部类id数组
    fields: list[FieldEntity]  # 字段列表
    methods: list[MethodEntity]  # 方法列表

    def __init__(self):
        self.ref_id = ''
        self.kind = ''
        self.prot = ''
        self.compound_name = ''
        self.base_compound_ref = []
        self.inner_class = []
        self.derived_compound_ref = []
        self.fields = []
        self.methods = []

    def print(self):
        print("ref_id: {0}, kind: {1}, compound_name: {2}".format(
            self.ref_id, self.kind, self.compound_name
        ))
        print("class fields: ----------------------")
        for f in self.fields:
            f.print()
        print("class methods: --------------------------")
        for m in self.methods:
            m.print()

    def set_class_info(self, ref_id='', kind='', prot='', compound_name=''):
        if not ref_id == '':
            self.ref_id = ref_id
        if not kind == '':
            self.kind = kind
        if not prot == '':
            self.prot = prot
        if not compound_name == '':
            self.compound_name = compound_name

    def add_inner_class(self, inner_class_id: str):
        self.inner_class.append(inner_class_id)

    def add_field(self, field: FieldEntity):
        self.fields.append(field)

    def add_method(self, method: MethodEntity):
        self.methods.append(method)

    def add_base_compound_ref(self, base_ref: str):
        self.base_compound_ref.append(base_ref)

    def add_derived_compound_ref(self, derived_ref: str):
        self.derived_compound_ref.append(derived_ref)

    def add_inner_class(self, inner: str):
        self.inner_class.append(inner)
