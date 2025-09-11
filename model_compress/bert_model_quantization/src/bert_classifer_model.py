#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：bert_classifer_model.py
@Create at   ：2025/9/11 15:55
@version     ：V1.0
@Author      ：erainm
@Description : bert分类模型
'''
from transformers import BertModel

from model_compress.bert_model_quantization.src.config import Config
import torch.nn as nn

conf = Config()

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(conf.bert_model_path)
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        out = self.fc(pooled)
        return out
