import pandas as pd

res1 = pd.read_pickle(f'./result-1.0')
res2 = pd.read_pickle(f'./result-2.0')
res3 = pd.read_pickle(f'./result-3.0')
res4 = pd.read_pickle(f'./result-4.0')

all_result = res1 + res2 + res3 + res4

def print_result(result, k):
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for minConf in s:
        count = 0
        print(f'minConf: {minConf}:')
        i = s.index(minConf)
        p, r, f = 0.0, 0.0, 0.0
        for res in result:
            if len(res) == 10 and type(res[i]) == list and len(res[i]) == 3:
                count += 1
                p += res[i][0]
                r += res[i][1]
                f += res[i][2]
        print(f'----------result of top {k}---{count}----\n'
              f'Precision: {p / count}, '
              f'Recall: {r / count}, '
              f'F1: {f / count}')


print(len(all_result))
print_result(all_result, 0)