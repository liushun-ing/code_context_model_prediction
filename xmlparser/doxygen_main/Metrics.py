from xmlparser.doxygen_main.ClassEntity import ClassEntity


class RepoMetrics:
    """
    解析的working periods的每个仓库的内容实体
    """
    repo_name: str
    repo_location: str
    classes: list[ClassEntity]

    def __init__(self):
        self.repo_name = ''
        self.repo_location = ''
        self.classes = []

    def print(self):
        print("repo_name: ", self.repo_name)
        print("repo_location: ", self.repo_location)
        for c in self.classes:
            c.print()

    def set_repo_info(self, repo_name: str, repo_location: str):
        self.repo_name = repo_name
        self.repo_location = repo_location

    def add_class_entity(self, class_entity: ClassEntity):
        self.classes.append(class_entity)

    def get_class_by_id(self, ref_id: str):
        for c in self.classes:
            if c.ref_id == ref_id:
                return c
        return None

    def is_inner_class(self, ref_id: str):
        for c in self.classes:
            for i in c.inner_class:
                if ref_id == i:
                    return True
        return False

    def get_field_by_id(self, ref_id: str):
        for c in self.classes:
            for f in c.fields:
                if f.ref_id == ref_id:
                    return f
        return None

    def get_method_by_id(self, ref_id: str):
        for c in self.classes:
            for f in c.methods:
                if f.ref_id == ref_id:
                    return f
        return None

    def get_element_by_id(self, ref_id: str):
        c = self.get_class_by_id(ref_id)
        if c is not None:
            return c
        m = self.get_method_by_id(ref_id)
        if m is not None:
            return m
        f = self.get_field_by_id(ref_id)
        if f is not None:
            return f
        return None
