#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：train.py
@Create at   ：2025/9/11 16:01
@version     ：V1.0
@Author      ：erainm
@Description : 训练bert模型
'''

import time
import warnings

import torch

from model_compress.bert_model_quantization.src.config import Config
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from model_compress.bert_model_quantization.src.bert_classifer_model import BertClassifier
from model_compress.bert_model_quantization.src.utils import build_dataloader

warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score

conf = Config()


def train_model():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。
    参数：
        无显式参数，所有配置通过全局 conf 对象获取。
    返回：
        无返回值，训练过程中保存最佳模型到指定路径。
    """
    # 第一步：加载训练、测试、验证数据集的DataLoader
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 第二步：定义训练参数
    device = conf.device
    num_epochs = conf.num_epochs
    # 第三步：初始化bert模型
    model = BertClassifier().to(device)
    # 第四步：定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    criterion = nn.CrossEntropyLoss()
    # 第五步：初始化最佳验证 F1分数，用于保存性能最好的模型
    best_dev_f1 = 0.0

    # 第六步：遍历每个轮次
    for epoch in range(num_epochs):
        # 设置模型为训练模式（启用电容屏out和batch norm）
        model.train()
        total_loss = 0 # 累计损失
        train_preds, train_labels = [], [] # 存储训练集预测和真实标签

        # 第七步：遍历批次，通过DataLoader进行模型训练
        for batch in tqdm(train_dataloader, desc=f"Bert Classifier Training Epoch {epoch + 1}/{num_epochs} ……"):
            # 7.1 提取批次数据并移动到设备上
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            # 7.2 前向传播：模型预测
            logits = model(input_ids, labels)
            # 7.3 损失计算
            loss = criterion(logits, labels)
            # 7.4 梯度归零
            optimizer.zero_grad()
            # 7.5 反向传播
            loss.backward()
            # 7.6 参数更新
            optimizer.step()

            # 7.7 累计损失计算
            total_loss += loss.item()
            # 7.8 获取预测结果（最大logits对应类别）
            preds = torch.argmax(logits, dim=1)
            # 7.9 存储预测和真实标签，用于计算训练集指标
            train_preds.extend(preds.tolist())
            train_labels.extend(labels.tolist())

            # 7.10 打印训练信息并评估验证集
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {total_loss/len(train_dataloader):.4f}")

            # 7.11 在验证集上评估模型
            report, f1score, accuracy, precision = model2dev(model, dev_dataloader, device)
            print(f"Dev F1: {f1score:.4f}")
            print(f"Dev Accuracy: {accuracy:.4f}")

            # 7.12 如果验证F1分数优于历史最佳，则保存模型
            if f1score > best_dev_f1:
                best_dev_f1 = f1score
                torch.save(model.state_dict(), conf.model_save_file)
        # 7.13 计算并打印训练集的分类报告
        train_report = classification_report(train_labels, train_preds, target_names=conf.class_list, output_dict=True)
        print("训练集分类报告 ---> ", train_report)



def model2dev(model, dataloader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能
    :param model:BERT 分类模型
    :param dataloader:数据加载器（验证或测试集）
    :param device:设备（"cuda" 或 "cpu"）
    :return:
        tuple: (分类报告, F1 分数, 准确度, 精确度)
            - report: 分类报告（包含每个类别的精确度、召回率、F1 分数等）。
            - f1score: 微平均 F1 分数。
            - accuracy: 准确度。
            - precision: 微平均精确度。
    """
    # 第一步：设置模型为评估模型（禁用dropout、batch norm）
    model.eval()

    # 第二步：初始化列表，存储预测结果和真实标签
    preds, true_labels = [], []

    # 第三步：禁用梯度计算以提高效率并减少内存使用
    with torch.no_grad():
        # 第四步：遍历数据加载器，逐批次进行预测
        for batch in tqdm(dataloader, desc="Bert Classifier Evaluating……"):
            # 4.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            # 4.2 前向传播：模型预测
            logits = model(input_ids, attention_mask)
            # 4.3 获取预测结果（最大logits对应类别）
            batch_preds = torch.argmax(logits, dim=1)
            # 4.4 存储预测和真实标签
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(batch_preds.cpu().numpy())

    # 第五步：计算分类报告、F1分数、准确度和精确度
    report = classification_report(true_labels, preds)
    f1score = f1_score(true_labels, preds, average='micro')
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='micro')

    return report, f1score, accuracy, precision


if __name__ == '__main__':
    train_model()

    # 1. 加载测试集数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 2. 初始化Bert分类模型
    model = BertClassifier()
    # 3. 加载预训练模型权重
    model.load_state_dict(torch.load(conf.model_save_file))
    # 4. 将模型移动到指定设备
    model.to(conf.device)
    # 5. 在测试集上评估模型
    test_report, f1score, accuracy, precision = model2dev(model, test_dataloader, conf.device)
    # 6. 打印测试集评估结果
    print("Test Set Evaluation:")
    print(f"Test F1: {f1score:.4f}")
    print("Test Classification Report:")
    print(test_report)