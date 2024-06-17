import numpy as np
from collections import Counter


def read_result(step: int, batch_index):
    result = []
    count = 0
    # with open(f'./origin_result/no_new_match_result_{step}.txt', 'r') as f:
    with open(f'./origin_result/no_new_match_result_{step}_seed.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('---') or line.startswith('node_id'):
                count += 1
                continue
            else:
                result.append(line.split(' '))
    print('total: ', count - 1)
    return result


def calculate_average_result(step):
    result = []
    curr = []
    with open(f'./origin_result/no_new_match_result_{step}_seed.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('---') or line.startswith('node_id'):
                if len(curr) > 0:
                    curr_result = []
                    # 计算当前的值
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        tp_count = len([item for item in curr if item[4] == "True" and float(item[2]) >= threshold])
                        fp_count = len([item for item in curr if item[4] == "False" and float(item[2]) >= threshold])
                        fn_count = len([item for item in curr if item[4] == "True" and float(item[2]) < threshold])
                        p = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                        r = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
                        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0
                        curr_result.append([p, r, f])
                    # all_pos = len([item for item in curr if item[4] == "True"])
                    # if all_pos == 2:
                    #     result.append(curr_result)
                    result.append(curr_result)
                else:
                    continue
                curr.clear()
            else:
                curr.append(line.split(' '))
    if len(curr) > 0:
        curr_result = []
        # 计算当前的值
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            tp_count = len([item for item in curr if item[4] == "True" and float(item[2]) >= threshold])
            fp_count = len([item for item in curr if item[4] == "False" and float(item[2]) >= threshold])
            fn_count = len([item for item in curr if item[4] == "True" and float(item[2]) < threshold])
            p = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            r = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0
            curr_result.append([p, r, f])
        # all_pos = len([item for item in curr if item[4] == "True"])
        # if all_pos == 2:
        #     result.append(curr_result)
        result.append(curr_result)
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    for minConf in s:
        i = s.index(minConf)
        p, r, f = 0.0, 0.0, 0.0
        nothing = 0
        for res in result:
            if res[i][0] == 0:
                nothing += 1
                continue
            p += res[i][0]
            r += res[i][1]
            f += res[i][2]
        p /= len(result)
        r /= len(result)
        print(f"{minConf:>10.1f} {p:>10.3f} {r:>10.3f} {2 * p * r / (p + r):>10.3f} nothing:{nothing}")


def true_positive(data):
    print('\n---------true positive----------\n')
    filtered_data = [item for item in data if item[4] == "True" and item[5] == "True"]
    print(len(filtered_data))

    # 提取confidence和stereotype
    third_column = [float(item[2]) for item in filtered_data]
    fourth_column = [item[3] for item in filtered_data]
    # fourth_column = [item[3] for item in filtered_data if 0 <= float(item[2]) <= 0.01]

    # 计算confidence数据的统计值
    third_column_min = np.min(third_column)
    third_column_max = np.max(third_column)
    third_column_median = np.median(third_column)
    third_column_mean = np.mean(third_column)
    third_column_variance = np.var(third_column)

    # confidence数据分段统计
    bins = np.arange(0.0, 1.01, 0.01)
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
        print(f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f} {hist[i]} {proportions[i]:.2%}")

    print("\nstereotype数据的统计:")
    sorted_labels = sorted(fourth_column_counts.items(), key=lambda item: item[1], reverse=True)
    for key, count in sorted_labels:
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    return third_column


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
        print(f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f} {hist[i]} {proportions[i]:.2%}")

    print("\nstereotype数据的统计:")
    sorted_labels = sorted(fourth_column_counts.items(), key=lambda item: item[1], reverse=True)
    for key, count in sorted_labels:
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    return third_column


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
    sorted_labels = sorted(fourth_column_counts.items(), key=lambda item: item[1], reverse=True)
    for key, count in sorted_labels:
        print(f"{key} {count} {fourth_column_proportions[key]:.2%}")
    return len(filtered_data)


def calculate_result(FPs, TPs, FN):
    # 阈值范围
    thresholds = np.arange(0.1, 1.1, 0.1)
    # 存储结果
    results = []
    for threshold in thresholds:
        tp_count = sum(1 for x in TPs if x >= threshold)
        fp_count = sum(1 for x in FPs if x >= threshold)
        fn_count = FN + len(TPs) - tp_count  # FN remains constant as it's the count of false negatives not the value

        # Avoid division by zero
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        results.append((threshold, tp_count, fp_count, fn_count, precision, recall, f1_score))

    # 打印结果
    print(f"\n{'Threshold':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    for threshold, tp_count, fp_count, fn_count, precision, recall, f1_score in results:
        print(
            f"{threshold:>10.1f} {tp_count:>5} {fp_count:>5} {fn_count:>5} {precision:>10.3f} {recall:>10.3f} {f1_score:>10.3f}")


if __name__ == '__main__':
    step = 1
    result = read_result(step, 0)
    # result = []
    # for i in range(27):
    #     result += read_result(step, i)
    print(len(result))
    FPs = false_positive(result)
    TPs = true_positive(result)
    FN = false_negative(result)
    # calculate_result(FPs, TPs, FN)
    calculate_average_result(step)
