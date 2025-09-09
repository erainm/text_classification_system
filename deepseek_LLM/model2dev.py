#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：model2dev.py
@Create at   ：2025/9/9 16:55
@version     ：V1.0
@Author      ：erainm
@Description : 模型评估
'''
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import warnings
warnings.filterwarnings("ignore")

from config import Config
conf = Config()

# 获取name2id
id2name = {}
with open(conf.class_datapath, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):  # class0  class 1
        label = line.strip()
        id2name[idx] = label
# print("类别映射:", id2name)
name2id = {v: k for k, v in id2name.items()}


def model2dev(dev_data):
    # 1. 初始化列表，存储预测结果和真实标签
    preds, true_labels = [], []

    start_time = time.time()

    # 2. 模型预测
    with open(dev_data, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="数据预测中....."):
            # 2.1 提取批次数据并移动到设备
            text, label = line.strip().split('\t')

            # 2.2 前向传播：模型预测
            result, _ = model2pred(text)
            pred = name2id[result]

            # 2.3 存储预测和真实标签
            preds.append(int(pred))
            true_labels.append(int(label))

    # 3. 计算分类报告、F1 分数、准确度和精确度
    accuracy = accuracy_score(true_labels, preds)  # 计算准确度
    precision = precision_score(true_labels, preds, average='micro')  # 使用微平均计算精确度
    f1score = f1_score(true_labels, preds, average='micro')  # 使用微平均计算 F1 分数
    report = classification_report(true_labels, preds)
    elapsed_time = (time.time() - start_time) * 1000
    print('预测完成！')

    # 4. 返回评估结果
    return accuracy, precision, recall_score, f1score, report, elapsed_time


if __name__ == '__main__':
    accuracy, precision, recall_score, f1score, report, elapsed_time = model2dev(conf.dev_datapath)
    print(f"*准确度:{accuracy}")
    print(f"*精确度:{precision}")
    print(f"*F1分数:{f1score}")
    print(f"*分类报告:{report}")
    print(f"*消耗时间:{elapsed_time:.2f}ms")