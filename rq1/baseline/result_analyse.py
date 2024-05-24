import numpy as np
from collections import Counter


def read_result(step: int):
    result = []
    with open(f'./origin_result/match_result_{step}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('---') or line.startswith('node_id'):
                continue
            else:
                result.append(line.split(' '))
    return result


def true_positive(data):
    print('\n---------true positive----------\n')
    filtered_data = [item for item in data if item[4] == "True" and item[5] == "True"]
    print(len(filtered_data))

    # 提取confidence和stereotype
    third_column = [float(item[2]) for item in filtered_data]
    fourth_column = [item[3] for item in filtered_data]

    # 计算confidence数据的统计值
    third_column_min = np.min(third_column)
    third_column_max = np.max(third_column)
    third_column_median = np.median(third_column)
    third_column_mean = np.mean(third_column)
    third_column_variance = np.var(third_column)

    # confidence数据分段统计
    bins = np.arange(0, 1.1, 0.1)
    hist, bin_edges = np.histogram(third_column, bins=bins)
    total_count = len(third_column)
    proportions = hist / total_count

    # 统计stereotype每个字符串出现的次数
    fourth_column_counts = Counter(fourth_column)
    total_fourth_count = len(fourth_column)
    fourth_column_proportions = {key: count / total_fourth_count for key, count in fourth_column_counts.items()}

    # 输出结果
    print("confidence数据的统计值:")
    print(f"最小值: {third_column_min}")
    print(f"最大值: {third_column_max}")
    print(f"中位数: {third_column_median}")
    print(f"平均值: {third_column_mean}")
    print(f"方差: {third_column_variance}")

    print("\nconfidence数据的分段统计 (0 到 1 间隔 0.1):")
    for i in range(len(hist)):
        print(f"{bin_edges[i]}-{bin_edges[i + 1]} {hist[i]} {proportions[i]:.2%}")

    print("\nstereotype数据的统计:")
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")


def false_positive(data):
    print('\n---------false positive----------\n')
    filtered_data = [item for item in data if item[4] == "False" and item[5] == "True"]
    print(len(filtered_data))

    # 提取confidence和stereotype
    third_column = [float(item[2]) for item in filtered_data]
    fourth_column = [item[3] for item in filtered_data]

    # 计算confidence数据的统计值
    third_column_min = np.min(third_column)
    third_column_max = np.max(third_column)
    third_column_median = np.median(third_column)
    third_column_mean = np.mean(third_column)
    third_column_variance = np.var(third_column)

    # confidence数据分段统计
    bins = np.arange(0, 1.1, 0.1)
    hist, bin_edges = np.histogram(third_column, bins=bins)
    total_count = len(third_column)
    proportions = hist / total_count

    # 统计stereotype每个字符串出现的次数
    fourth_column_counts = Counter(fourth_column)
    total_fourth_count = len(fourth_column)
    fourth_column_proportions = {key: count / total_fourth_count for key, count in fourth_column_counts.items()}

    # 输出结果
    print("confidence数据的统计值:")
    print(f"最小值: {third_column_min}")
    print(f"最大值: {third_column_max}")
    print(f"中位数: {third_column_median}")
    print(f"平均值: {third_column_mean}")
    print(f"方差: {third_column_variance}")

    print("\nconfidence数据的分段统计 (0 到 1 间隔 0.1):")
    for i in range(len(hist)):
        print(f"{bin_edges[i]}-{bin_edges[i + 1]} {hist[i]} {proportions[i]:.2%}")

    print("\nstereotype数据的统计:")
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")


def false_negative(data):
    print('\n---------false negative----------\n')
    # 筛选第五列等于 "True" 且第六列等于 "False" 的数据
    filtered_data = [item for item in data if item[4] == "True" and item[5] == "False"]
    print(len(filtered_data))

    # 提取stereotype
    fourth_column = [item[3] for item in filtered_data]

    # 统计stereotype每个字符串出现的次数
    fourth_column_counts = Counter(fourth_column)
    total_fourth_count = len(fourth_column)
    fourth_column_proportions = {key: count / total_fourth_count for key, count in fourth_column_counts.items()}

    print("\nstereotype数据的统计:")
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")


if __name__ == '__main__':
    step = 2
    result = read_result(step)
    print(len(result))
    false_positive(result)
    false_negative(result)
    true_positive(result)
