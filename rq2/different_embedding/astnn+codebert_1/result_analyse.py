import numpy as np
from collections import Counter


def read_result(step: int):
    result = []
    with open(f'./specific_result_{step}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('---'):
                continue
            else:
                result.append(line.split(' '))
    return result


def all_positive(data):
    print('\n---------all positive----------\n')
    filtered_data = [item for item in data if item[4] == "True"]
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
    bins = np.arange(0.0, 1.1, 0.1)
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
        print(f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}: {hist[i]} {proportions[i]:.2%}")

    print("\nkind 数据的统计:")
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    return third_column


def true_node(data):
    print('\n---------true nodes----------\n')
    filtered_data = [item for item in data if item[4] == "True"]
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
        print(f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f} {hist[i]} {proportions[i]:.2%}")

    print("\nkind 数据的统计:")
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    return third_column


def false_node(data):
    print('\n---------false nodes----------\n')
    # 筛选第五列等于 "True" 且第六列等于 "False" 的数据
    filtered_data = [item for item in data if item[4] == "False"]
    print(len(filtered_data))

    # 提取stereotype
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
        print(f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}: {hist[i]} {proportions[i]:.2%}")

    print("\nkind 数据的统计:")
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    return third_column


def specific_analyze(result, threshold):
    print(f'\n---------threshold: {threshold}----------')
    print('---------false negative----------')
    kind = [item[3] for item in result if item[4] == "True" and float(item[2]) < threshold]
    print(len(kind))
    # 统计stereotype每个字符串出现的次数
    fourth_column_counts = Counter(kind)
    total_fourth_count = len(kind)
    fourth_column_proportions = {key: count / total_fourth_count for key, count in fourth_column_counts.items()}
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    print('---------false positive----------')
    kind = [item[3] for item in result if item[4] == "False" and float(item[2]) >= threshold]
    print(len(kind))
    # 统计stereotype每个字符串出现的次数
    fourth_column_counts = Counter(kind)
    total_fourth_count = len(kind)
    fourth_column_proportions = {key: count / total_fourth_count for key, count in fourth_column_counts.items()}
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    print('---------true positive----------')
    kind = [item[3] for item in result if item[4] == "True" and float(item[2]) >= threshold]
    print(len(kind))
    # 统计stereotype每个字符串出现的次数
    fourth_column_counts = Counter(kind)
    total_fourth_count = len(kind)
    fourth_column_proportions = {key: count / total_fourth_count for key, count in fourth_column_counts.items()}
    for key, count in fourth_column_counts.items():
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")



def calculate_result(true_nodes, false_nodes):
    # 阈值范围
    thresholds = np.arange(0.1, 1.1, 0.1)
    # 存储结果
    results = []
    for threshold in thresholds:
        tp_count = sum(1 for x in true_nodes if x >= threshold)
        fp_count = sum(1 for x in false_nodes if x >= threshold)
        fn_count = sum(1 for x in true_nodes if x < threshold)
        # Avoid division by zero
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        results.append((threshold, tp_count, fp_count, fn_count, precision, recall, f1_score))
    # 打印结果
    print(f"{'Threshold':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    for threshold, tp_count, fp_count, fn_count, precision, recall, f1_score in results:
        print(
            f"{threshold:>10.1f} {tp_count:>5} {fp_count:>5} {fn_count:>5} {precision:>10.3f} {recall:>10.3f} {f1_score:>10.3f}")


if __name__ == '__main__':
    step = 1
    threshold = 0.5
    result = read_result(step)
    print(len(result))
    false_nodes = false_node(result)
    true_nodes = true_node(result)
    specific_analyze(result, threshold)
    print(f"\nAll nodes: {len(false_nodes) + len(true_nodes)}")
    calculate_result(true_nodes, false_nodes)
