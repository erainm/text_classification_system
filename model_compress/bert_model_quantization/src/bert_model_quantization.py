#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：bert_model_quantization.py
@Create at   ：2025/9/11 15:20
@version     ：V1.0
@Author      ：erainm
@Description : 基于bert构建的分类模型进行量化
'''
from bert_classifer_model import BertClassifier
from config import Config
import torch
from utils import build_dataloader
from train import model2dev

if __name__ == '__main__':
    conf = Config()
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 加载模型
    device = conf.device
    model = BertClassifier()
    model_file = conf.model_save_file
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    print("量化前模型结构 ----------- > \n", model)

    # 动态量化Bert模型
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("量化后模型结构 ----------- > \n", quantized_model)

    # model2dev 测试量化后的模型
    report, f1score, accuracy, precision = model2dev(quantized_model, test_dataloader, device)
    print("Test Classification Report:", report)
    print("Test F1:", f1score)
    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)

    # 计算8-bit量化后模型的内存占用
    # sum(p.numel() * p.element_size() for p in quantized_model.parameters()): 遍历模型参数，计算每个参数张量的元素总数（numel）乘以每个元素字节大小（element_size），累加得到总字节数
    # / 1024 ** 2: 将字节数转换为兆字节（MB）
    # :.2f: 保留两位小数
    print(f"8-bit 量化后的模型内存: {sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 ** 2:.2f} MB")

