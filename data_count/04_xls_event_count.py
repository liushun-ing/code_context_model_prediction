# 对过滤后的events进行数据统计
import os
from datetime import datetime
from os.path import isdir, join
import numpy as np
import xlrd
import xlwt

write_excel = None
new_sheet = None
line_index = None
row0 = ["bug report", "Interaction Event", "average_event_duration(/s)"]


def init_xsl():
    global write_excel
    write_excel = xlwt.Workbook()  # 创建工作表


def create_new_sheet(sheet_name: str):
    global new_sheet
    global line_index
    global row0
    line_index = 0
    new_sheet = write_excel.add_sheet(sheet_name, cell_overwrite_ok=True)  # 创建sheet
    # 生成第一行
    for i in range(0, len(row0)):
        new_sheet.write(line_index, i, row0[i])


def save_xsl():
    global write_excel
    file_name = "filtered_event_data.xls"
    if os.path.exists(file_name):
        os.remove(file_name)
    # 保存文件
    write_excel.save(file_name)


def write_xsl(row: list):
    global line_index
    global new_sheet
    global row0
    line_index += 1
    # 生成第一行
    for i in range(0, len(row0)):
        new_sheet.write(line_index, i, row[i])


def read_xls_excel(url, index=1):
    """
    读取xls格式文件
    参数：
        url:文件路径
        index：工作表序号（第几个工作表，传入参数从1开始数）
    返回：
        data:表格中的数据
    """
    # 打开指定的工作簿
    workbook = xlrd.open_workbook(url)
    # 获取工作簿中的所有表格
    sheets = workbook.sheet_names()
    # 获取工作簿中所有表格中的的第 index 个表格
    worksheet = workbook.sheet_by_name(sheets[index - 1])
    # 定义列表存储表格数据
    data = []
    # 遍历每一行数据
    for i in range(0, worksheet.nrows):
        # 定义表格存储每一行数据
        da = []
        # 遍历每一列数据
        for j in range(0, worksheet.ncols):
            # 将行数据存储到da列表
            da.append(worksheet.cell_value(i, j))
        # 存储每一行数据
        data.append(da)
    # 返回数据
    return data


def get_duration_of_event(date_list: list[str]):
    date_list.sort()
    duration_list = []  # 相差的秒数
    for i in range(len(date_list) - 1):
        time1 = datetime.strptime(date_list[i], '%Y-%m-%d %H:%M:%S')
        time2 = datetime.strptime(date_list[i + 1], '%Y-%m-%d %H:%M:%S')
        duration = time2 - time1
        duration_list.append(duration.total_seconds())
    return 0 if len(duration_list) == 1 else sum(duration_list) / (len(duration_list) - 1)


file_path = os.path.dirname(os.path.realpath(__file__))
file_dir_list = os.listdir(file_path)
init_xsl()
for file_dir in file_dir_list:
    project_path = join(file_path, file_dir)
    if not isdir(project_path) or file_dir == '__pycache__':
        continue
    print("current project:" + file_dir)
    create_new_sheet(file_dir)
    xls_list = os.listdir(project_path)
    # 进入bug目录
    for xls_file in sorted(xls_list, key=len):
        xls_path = join(project_path, xls_file)
        xls_data = read_xls_excel(xls_path)  # 拿到所有的data
        # 下面需要统计每个 bug 的 event 个数和时间间隔的平均值
        a = np.array(xls_data)
        event_duration = get_duration_of_event(a[1:, -1])
        print("{} bug has {} event, duration is {}".format(xls_file, len(xls_data), event_duration))
        write_xsl([xls_file, len(xls_data), event_duration])
save_xsl()

