import os

import embedding
import train_BPE


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU编号

def train_mylyn(step, r):
    # 训练word2vec模型
    train_BPE.main_func(step=step, description='mylyn', r=r)
    # 转换为词向量
    embedding.main_func(step=step, description='mylyn', r=r)


# 全部训练
def train_all(step, r):
    """
    全部训练,得到节点嵌入
    """
    # 训练word2vec模型
    # train_BPE.main_func(step=step, description='all', r=r)
    # 转换为词向量
    embedding.main_func(step=step, description='all', r=r)


def train_onlymylyn(step):
    """
    只把mylyn作为训练数据,得到节点嵌入
    """
    # train_word2vec.main_func(step=step, description='onlymylyn')
    embedding.main_func(step=step, description='onlymylyn')


def train_nopde(step):
    """
    把除pde外的三个项目作为训练数据,得到节点嵌入
    """
    train_BPE.main_func(step=step, description='nopde')
    embedding.main_func(step=step, description='nopde')


def train_noplatform(step):
    """
    把除 platform 外的三个项目作为训练数据,得到节点嵌入
    """
    train_BPE.main_func(step=step, description='noplatform')
    embedding.main_func(step=step, description='noplatform')


# train_mylyn(step=1, r=0.84)
# train_mylyn(step=2, r=0.84)
# train_mylyn(step=3, r=0.84)
# train_mymylyn(step=1, r=0.8)
# train_mymylyn(step=2, r=0.8)
# train_mymylyn(step=3, r=0.8)
train_all(step=1, r=0.8)  # pde=55
# train_all(step=2, r=0.8) # 365
# train_all(step=3, r=0.8)
# train_onlymylyn(step=1)
# train_onlymylyn(step=2) #
# train_onlymylyn(step=3)  # 0
