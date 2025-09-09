#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：predict.py
@Create at   ：2025/9/9 09:40
@version     ：V1.0
@Author      ：erainm
@Description : 预测模型
'''
import torch
from transformers import BertTokenizer
from bert_classifer_model import BertClassifier
from config import Config
# 初始化配置
conf = Config()
device = conf.device
tokenizer = conf.tokenizer
model = BertClassifier().to(device)
model.load_state_dict(torch.load("../save_models/bert20250521_.pt"))
model.eval()

#预测函数
def predict(data):
    # 1.获取数据 data['text']
    text = data['text']

    # 2.预处理数据----text===> input_ids,attention_mask <=== tokenizer.encode_plus
    tokenize = tokenizer.encode_plus(text, return_tensors='pt')
    input_ids = tokenize['input_ids'].to(device)
    attention_mask = tokenize['attention_mask'].to(device)

    # 3.模型预测
    ## 3.1 关闭梯度计算
    with torch.no_grad():
        ## 3.2 前向推理 model()
        pred_logits = model(input_ids, attention_mask)
        ## 3.3 获取预测结果---softmax  argmax(最大概率值所对应的索引)  class_name
        pred_prob = torch.softmax(pred_logits, dim=1)
        pred_ids = torch.argmax(pred_prob, dim=1)
        pred_class = conf.class_list[pred_ids]

    return {"text": text, "pred_class": pred_class}