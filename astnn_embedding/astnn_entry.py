import os

import embedding
import train_word2vec


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU编号

def train_mylyn(step, r, hidden_dim, code_dim):
    # 训练word2vec模型
    train_word2vec.main_func(step=step, description='mylyn', r=r, code_dim=code_dim)
    # 转换为词向量
    embedding.main_func(step=step, description='mylyn', r=r, hidden_dim=hidden_dim, code_dim=code_dim)


# 全部训练
def train_all(step, r, hidden_dim, code_dim):
    """
    全部训练,得到节点嵌入
    """
    # 训练word2vec模型
    train_word2vec.main_func(step=step, description='all', r=r, code_dim=code_dim)
    # 转换为词向量
    embedding.main_func(step=step, description='all', r=r, hidden_dim=hidden_dim, code_dim=code_dim)


def train_onlymylyn(step, hidden_dim, code_dim):
    """
    只把mylyn作为训练数据,得到节点嵌入
    """
    train_word2vec.main_func(step=step, description='onlymylyn', code_dim=code_dim)
    embedding.main_func(step=step, description='onlymylyn', hidden_dim=hidden_dim, code_dim=code_dim)


def train_nopde(step, hidden_dim, code_dim):
    """
    把除pde外的三个项目作为训练数据,得到节点嵌入
    """
    train_word2vec.main_func(step=step, description='nopde', code_dim=code_dim)
    embedding.main_func(step=step, description='nopde', hidden_dim=hidden_dim, code_dim=code_dim)


def train_noplatform(step, hidden_dim, code_dim):
    """
    把除 platform 外的三个项目作为训练数据,得到节点嵌入
    """
    train_word2vec.main_func(step=step, description='noplatform', code_dim=code_dim)
    embedding.main_func(step=step, description='noplatform', hidden_dim=hidden_dim, code_dim=code_dim)


if __name__ == '__main__':
    # train_mylyn(step=1, r=0.8, hidden_dim=256, code_dim=256)
    # train_mylyn(step=2, r=0.8, hidden_dim=256, code_dim=256)
    # train_mylyn(step=3, r=0.8, hidden_dim=256, code_dim=256)
    train_onlymylyn(1, hidden_dim=256, code_dim=256)
    train_onlymylyn(2, hidden_dim=256, code_dim=256)
    train_onlymylyn(3, hidden_dim=256, code_dim=256)
