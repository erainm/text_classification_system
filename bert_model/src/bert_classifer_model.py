#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：bert_classifer_model.py
@Create at   ：2025/9/9 09:39
@version     ：V1.0
@Author      ：erainm
@Description : 构建bert模型
'''
import torch.nn as nn
from transformers import BertModel

from bert_model.src.config import Config

conf = Config()

class BertClassifier(nn.Module):
    # Bert + 全连接层的分类模型
    def __init__(self):
        """
        初始化模型，包括Bert模型和全连接层
        """
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        out = self.fc(pooled)
        return out
