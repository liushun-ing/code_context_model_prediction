from decimal import Decimal

import pandas as pd

all_result = []
step = 3
batch = [0, 27, 14, 10]
for batch_index in range(batch[step]):
    batch_result = pd.read_pickle(f'./origin_result/no_result_full_{step}_batch_{batch_index}.pkl')
    all_result += batch_result

print(len(all_result))

s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
for minConf in s:
    i = s.index(minConf)
    p, r, f = 0.0, 0.0, 0.0
    for res in all_result:
        p += res[i][0]
        r += res[i][1]
        f += res[i][2]
    p = Decimal(p / len(all_result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    r = Decimal(r / len(all_result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    # f = Decimal(f / len(all_result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    f = Decimal(2 * p * r / (p + r)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    print(f"{minConf:>10.1f} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

result = []
for batch_index in range(batch[step]):
    curr = []
    with open(f'./origin_result/no_new_match_result_{step}_batch_{batch_index}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('---') or line.startswith('node_id'):
                if len(curr) > 0:
                    curr_result = []
                    # 计算当前的值
                    for k in [1, 3, 5]:
                        # 选择最高的 k 个
                        # 过滤出第二列大于0的行
                        filtered_data = [row for row in curr if float(row[2]) > 0]
                        # 按第二列数据从大到小排序
                        sorted_data = sorted(filtered_data, key=lambda x: x[2], reverse=True)
                        # 获取前k行数据
                        top_k_data = sorted_data[:k]
                        all_true_count = len([item for item in curr if item[4] == "True"])
                        true_count = len([item for item in top_k_data if item[4] == "True"])
                        false_count = len([item for item in top_k_data if item[4] == "False"])
                        p = true_count / k
                        r = true_count / all_true_count if all_true_count > 0 else 0
                        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0
                        curr_result.append([p, r, f])
                    result.append(curr_result)
                else:
                    continue
                curr.clear()
            else:
                curr.append(line.split(' '))
    if len(curr) > 0:
        curr_result = []
        # 计算当前的值
        for k in [1, 3, 5]:
            # 选择最高的 k 个
            # 过滤出第二列大于0的行
            filtered_data = [row for row in curr if float(row[2]) > 0]
            # 按第二列数据从大到小排序
            sorted_data = sorted(filtered_data, key=lambda x: x[2], reverse=True)
            # 获取前k行数据
            top_k_data = sorted_data[:k]
            all_true_count = len([item for item in curr if item[4] == "True"])
            true_count = len([item for item in top_k_data if item[4] == "True"])
            # false_count = len([item for item in top_k_data if item[4] == "False"])
            p = true_count / k
            r = true_count / all_true_count if all_true_count > 0 else 0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0
            curr_result.append([p, r, f])
        result.append(curr_result)
s = [1, 3, 5]
print(f"{'top-k':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
for k in s:
    i = s.index(k)
    p, r, f = 0.0, 0.0, 0.0
    for res in result:
        p += res[i][0]
        r += res[i][1]
        f += res[i][2]
    p /= len(result)
    r /= len(result)
    print(f"{k:>10.1f} {p:>10.3f} {r:>10.3f} {2 * p * r / (p + r):>10.3f}")

