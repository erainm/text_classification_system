#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：config.py
@Create at   ：2025/9/11 10:47
@version     ：V1.0
@Author      ：erainm
@Description : bert模型增加量化后的配置文件
'''
import datetime

from transformers import BertModel, BertTokenizer, BertConfig

current_date=datetime.datetime.now().date().strftime("%Y%m%d")

class Config(object):
    def __init__(self):
        # 模型名词
        self.model_name = "bert"
        # 数据集路径
        self.data_path = "../../../data"
        self.train_file = self.data_path + "/train.txt"
        self.test_file = self.data_path + "/test.txt"
        self.dev_file = self.data_path + "/dev.txt"
        # 类别文件
        self.class_file = self.data_path + "/class.txt"
        self.class_list = [line.strip() for line in open(self.class_file, "r", encoding="utf-8")]

        # 模型保存
        self.model_save_path = "../save_models"
        # 模型训练结果保存位置
        self.model_save_file = self.model_save_path + f"/bert_{current_date}.pt"
        # 量化模型保存位置
        self.quantized_model_save_file = self.model_save_path + f"quantized_bertclassifer_model_{current_date}.pt"

        # 模型训练 + 预测，量化时启用cpu
        self.device = "cpu"

        self.num_classes = len(self.class_list)
        self.num_epochs = 2
        self.batch_size = 2
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_model_path = "../../../bert_model/bert-base-chinese"
        self.bert_model = BertModel.from_pretrained(self.bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_model_path)
        self.hidden_size = 768

if __name__ == '__main__':
    conf = Config()
    print(conf.bert_config)
    input_size = conf.tokenizer.convert_tokens_to_ids(["你", "好", "中国", "人"])
    print(input_size)
    print(conf.class_list)


