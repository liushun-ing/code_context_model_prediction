from decimal import Decimal

import pandas as pd


def print_result(result, k):
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for minConf in s:
        print(f'minConf: {minConf}:')
        i = s.index(minConf)
        p, r, f = 0.0, 0.0, 0.0
        for res in result:
            p += res[i][0]
            r += res[i][1]
            f += res[i][2]
        p = Decimal(p / len(result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        r = Decimal(r / len(result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        f = Decimal(f / len(result)).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        print(f'----------result of top {k}-------\n'
              f'Precision: {p}, '
              f'Recall: {r}, '
              f'F1: {f}')


step = 1
all_result = []
for batch_index in range(13):
    res = pd.read_pickle(f'./origin_result/result_full_{step}_{batch_index}.pkl')
    all_result = all_result + res
print(len(all_result))
print_result(all_result, 0)
