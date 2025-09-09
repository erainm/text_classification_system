#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：config.py
@Create at   ：2025/9/9 09:37
@version     ：V1.0
@Author      ：erainm
@Description : 配置文件（包含模型和训练需要的各种参数）
'''
import os
# 限制 transformers 的功能，避免加载不需要的依赖
os.environ['TRANSFORMERS_NO_PYARROW'] = '1'
# 避免加载 pyarrow
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'

import torch
import datetime
from transformers import BertModel, BertTokenizer, BertConfig

current_date = datetime.datetime.now().date().strftime("%Y%m%d")

class Config(object):
    def __init__(self):
        self.model_name = "bert"
        self.data_root_path = "../../data"
        self.train_file = self.data_root_path + "/train.txt" # 训练集
        self.test_file = self.data_root_path + "/test.txt" # 测试集
        self.dev_file = self.data_root_path + "/dev.txt" # 验证集

        self.class_file = self.data_root_path + "/class.txt" # 类别文件
        self.class_list = [line.strip() for line in open(self.class_file, encoding="utf-8")] # 类别列表

        self.save_model_path = "../save_model/train_bertclassifer_model.pt"

        # 设备有GPU则用GPU没有就用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = len(self.class_list) # 类别数
        self.num_epochs = 2 #epoch数
        self.batch_size = 32 # mini-batch大小
        self.pad_size = 32 # 每句话处理成的长度
        self.learning_rate = 5e-5 # 学习率
        self.bert_path = "../bert-base-chinese" # 预训练BERT模型路径
        self.bert_model = BertModel.from_pretrained(self.bert_path) # bert模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # bert模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path) # bert模型的配置
        self.hidden_size = 768 # bert模型隐藏层大小

if __name__ == '__main__':
    conf = Config()
    print("bert_config ---> \n", conf.bert_config)
    input_size = conf.tokenizer.convert_tokens_to_ids(["你", "好", "中国", "人"])
    print("input_size ---> ", input_size)
    print("class_list ---> ", conf.class_list)
