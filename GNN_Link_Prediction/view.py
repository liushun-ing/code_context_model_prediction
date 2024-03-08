import os
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt

gnn_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'GNN_Link_Prediction')

df = pd.read_csv(join(gnn_path, 'model_1', 'gnn_result.tsv'))

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
plt.show()
