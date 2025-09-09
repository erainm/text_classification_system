#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：train.py
@Create at   ：2025/9/9 09:40
@version     ：V1.0
@Author      ：erainm
@Description : 训练模型
'''
import torch
from fasttext_pybind import f1score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from torch.optim import AdamW
import torch.nn as nn

from baseline_model_random_forest.rf_predict import report
from bert_model.src.bert_classifer_model import BertClassifier
from bert_model.src.utils import build_dataloader
from config import Config
from tqdm import tqdm

conf = Config()
def train_model():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。
    参数：
        无显式参数，所有配置通过全局 conf 对象获取。
    返回：
        无返回值，训练过程中保存最佳模型到指定路径。
    """
    # 第一步：加载训练、测试和验证数据集的dataloader
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 第二步：定义训练参数，从配置中获取
    device = conf.device
    num_epochs = conf.num_epochs

    # 第三步：初始化Bert分类模型
    model = BertClassifier().to(device)
    # 第四步：定义优化器（AdamW，适合Transformer模型）和损失函数（交叉熵）
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 第五步：初始化最佳验证F1分数，用户保存性能最好的模型
    best_dev_f1 = 0.0

    # 第六步：遍历每个训练轮次
    for epoch in range(num_epochs):
        # 设置模型为训练模式（启用dropout和batch norm）
        model.train()
        total_loss = 0
        train_preds, train_labels = [], [] # 用于存储训练集预测和真实标签

        # 第七步：遍历训练 DataLoader 进行模型训练(每个批次)
        for batch in tqdm(train_dataloader, desc=f"Bert Classifier Training Epoch {epoch + 1} / {num_epochs} …"):
            # 获取批次数据，并将数据移动到设备（cuda或cpu）
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # 7.1 前向传播：模型预测
            logits = model(input_ids, attention_mask)
            # 7.2 损失计算
            loss = criterion(logits, labels)
            # 7.3 梯度归零
            optimizer.zero_grad()
            # 7.4 反向传播
            loss.backward()
            # 7.5 参数更新
            optimizer.step()

            # 累计损失
            total_loss += loss.item()
            # 获取最大预测结果（最大 logits 对应类别）
            preds = torch.argmax(logits, dim=1)
            # 存储预测和真实标签，用于计算训练指标
            train_preds.extend(preds.tolist())
            train_labels.extend(labels.tolist())

            # 打印训练信息并评估验证
            print(f"Epoch: {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {total_loss / len(train_dataloader):.4f}")

            # 在验证集上评估
            report, f1score, accuracy, precision = model2dev(model, dev_dataloader, device)
            print(f"Dev F1: {f1score:.4f}")
            print(f"Dev Accuracy: {accuracy:.4f}")

            # 如果验证f1分数优于历史最佳，保存模型
            if f1score > best_dev_f1:
                best_dev_f1 = f1score
                torch.save(model.state_dict(), conf.save_model_path)
                print("模型保存成功~~")
        train_report = classification_report(train_labels, train_preds, target_names=conf.class_list, output_dict=True)
        print(train_report)


def model2dev(model, data_loader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能。

    参数：
        model (nn.Module): BERT 分类模型。
        data_loader (DataLoader): 数据加载器（验证或测试集）。
        device (str): 设备（"cuda" 或 "cpu"）。

    返回：
        tuple: (分类报告, F1 分数, 准确度, 精确度)
            - report: 分类报告（包含每个类别的精确度、召回率、F1 分数等）。
            - f1score: 微平均 F1 分数。
            - accuracy: 准确度。
            - precision: 微平均精确度。
    """
    # 第一步：设置模型为评估模式
    model.eval()
    # 第二步：初始化列表，存储预测结果和真实标签
    preds, true_labels = [],[]
    # 第三步： 禁用梯度计算（提高效率和减少内存占用）
    with torch.no_grad():
        # 第四步：遍历数据加载器，逐批次进行预测
        for batch in tqdm(data_loader, desc="Bert Classifier Evaluating……"):
            # 4.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # 4.2 前向传播：模型预测
            logits = model(input_ids, attention_mask)

            # 4.3 获取最大预测结果（最大 logits 对应的类别）
            batch_preds = torch.argmax(logits, dim=1)

            # 4.4 存储预测和真实标签
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # 第五步：分类报告、F1分数、准确率和精确度
    report = classification_report(true_labels, preds)
    f1score = f1_score(true_labels, preds, average='micro') # 使用微平均计算 F1 分数
    accuracy = accuracy_score(true_labels, preds) # 计算精确度
    precision = precision_score(true_labels, preds, average='micro') # 使用微平均计算精确度

    return report, f1score, accuracy, precision


if __name__ == '__main__':
    # 主程序入口
    train_model()

    # 1. 加载测试集数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 2. 初始化 BERT 分类模型
    model = BertClassifier()
    # 3. 加载预训练模型权重
    model.load_state_dict(torch.load(conf.save_model_path))
    # 4. 将模型移动到指定设备
    model.to(conf.device)
    # 5. 在测试集上评估模型
    test_report, f1score, accuracy, precision = model2dev(model, test_dataloader, conf.device)
    # 6. 打印测试集评估结果
    print("Test Set Evaluation:")
    print(f"Test F1: {f1score:.4f}")
    print("Test Classification Report:")
    print(test_report)