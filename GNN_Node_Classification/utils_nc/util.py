import os
from os.path import join
from typing import Literal

import pandas as pd
import torch

STEP = Literal[1, 2, 3]


def save_result(result, root_path, step: STEP, save_name):
    """
    保存模型在验证集上的训练结果

    :param result: 训练数据结果
    :param root_path: 需要保存的根目录
    :param step: 模式，也就是步长，1，2，3
    :param save_name: 保存结果的名字
    :return: none
    """
    model_path = join(root_path, 'model_' + str(step))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    result_path = join(model_path, f'{save_name}.tsv')
    df = pd.DataFrame(result,
                      columns=['epoch', 'Train Loss', 'Train Accuracy', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1',
                               'AUROC'])
    df.to_csv(result_path, index=False)
    print(f'result saved to {result_path}')


def save_model(model, root_path, step: STEP, save_name):
    """
    保存训练好的模型

    :param model: 模型
    :param root_path: 需要保存的根目录
    :param step: 模式，也就是步长，1，2，3
    :param save_name: 模型名字
    :return: none
    """
    model_path = join(root_path, 'model_' + str(step))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = join(model_path, f'{save_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


def load_model(model, root_path, step: STEP, load_name):
    """
    加载训练好的模型 \n
    loaded_model = GATModel() \n
    loaded_model = load_model(loaded_model, model_path) \n
    loaded_model = loaded_model.to(device)

    :param model: 模型
    :param root_path: 需要保存的根目录
    :param step: 模式，也就是步长，1，2，3
    :param load_name: 模型名字
    :return: none
    """
    model_path = join(root_path, 'model_' + str(step))
    model_path = join(model_path, f'{load_name}.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')
    else:
        print('no model loaded')
    return model


def threshold_tensor(input_tensor, threshold=0.5):
    """
    将小数张量进行阈值处理，大于阈值的元素变成 1，小于等于阈值的元素变成 0。

    :param input_tensor: 输入的小数张量
    :type input_tensor: torch.Tensor
    :param threshold: 阈值，默认为0.5
    :type threshold: float
    :return: 处理后的二值张量
    :rtype: torch.Tensor
    """
    binary_tensor = torch.where(input_tensor > threshold, 1.0, 0.0)
    return binary_tensor


def count_equal_elements(tensor1, tensor2):
    """
    计算两个张量中相等元素的数量。

    :param tensor1: 第一个张量
    :type tensor1: torch.Tensor
    :param tensor2: 第二个张量
    :type tensor2: torch.Tensor
    :return: 相等元素的数量
    :rtype: int
    """
    equal_elements = torch.eq(tensor1, tensor2)
    count = torch.sum(equal_elements).item()
    return count


def count_elements_equal_to_value(tensor1, tensor2, value):
    """
    计算两个张量中等于特定值的相等元素的数量。

    :param tensor1: 第一个张量
    :type tensor1: torch.Tensor
    :param tensor2: 第二个张量
    :type tensor2: torch.Tensor
    :param value: 特定的值
    :type value: float
    :return: 等于特定值的相等元素的数量
    :rtype: int
    """
    equal_to_value = (tensor1 == value) & (tensor2 == value)
    count = torch.sum(equal_to_value).item()
    return count


def count_values_equal_to(tensor, value):
    """
    统计张量中等于特定值的元素的数量。

    :param tensor: 输入的张量
    :type tensor: torch.Tensor
    :param value: 特定的值
    :type value: float
    :return: 等于特定值的元素的数量
    :rtype: int
    """
    equal_to_value = torch.eq(tensor, value)
    count = torch.sum(equal_to_value).item()
    return count
