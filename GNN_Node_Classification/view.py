from os.path import join

import pandas as pd
import matplotlib.pyplot as plt


def main_func(result_path, step, load_name):
    df = pd.read_csv(join(result_path, f'model_{str(step)}', f'{load_name}.tsv'))
    plt.plot(df['epoch'].tolist(), df['Train Loss'].tolist(), '.--', label='Train Loss')
    plt.plot(df['epoch'].tolist(), df['Train Accuracy'].tolist(), '.--', label='Train Accuracy')
    plt.plot(df['epoch'].tolist(), df['Loss'].tolist(), '.--', label='Loss')
    plt.plot(df['epoch'].tolist(), df['Accuracy'].tolist(), '.--', label='Accuracy')
    plt.plot(df['epoch'].tolist(), df['Precision'].tolist(), '.--', label='Precision')
    plt.plot(df['epoch'].tolist(), df['Recall'].tolist(), '.--', label='Recall')
    plt.plot(df['epoch'].tolist(), df['F1'].tolist(), '.--', label='F1')
    plt.plot(df['epoch'].tolist(), df['AUROC'].tolist(), '.--', label='AUROC')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)  # 显示上面的label
    plt.xlabel('epoch')  # x_label
    plt.tight_layout()
    plt.title(load_name)
    plt.show()
