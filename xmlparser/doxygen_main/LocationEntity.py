class LocationEntity:
    """
    字段和方法的location实体
    """

    file = ''  # 所在file路径

    line = -1  # 应该是声明字段或者方法的行

    column = -1  # 字段名或方法名的列，从返回值开始位置算
    """从返回值位置开始列下标"""

    body_file = ''  # 方法体的file
    """方法体文件"""

    body_start = -1  # 方法体开始行
    """方法体开始行"""

    body_end = -1  # 方法体结束行
    """方法体结束行"""

    def __init__(self, file='', line=-1, column=-1, body_file='', body_start=-1, body_end=-1):
        self.file = file
        self.line = line
        self.column = column
        self.body_file = body_file
        self.body_start = body_start
        self.body_end = body_end

    def print(self):
        print("file: {0}, line: {1}, column: {2}, body_file: {3}, body_start: {4}, body_end: {5}".format(
            self.file, self.line, self.column, self.body_file, self.body_start, self.body_end)
        )
