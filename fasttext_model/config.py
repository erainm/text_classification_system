#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：config.py
@Create at   ：2025/9/5 15:04
@version     ：V1.0
@Author      ：erainm
@Description : 配置文件，主要用于路劲配置
'''
class Config(object):
    def __init__(self):
        # 原始数据路径
        self.train_datapath = "./data/train.txt"
        self.test_datapath = "./data/test.txt"
        self.dev_datapath = "./data/dev.txt"
        self.dev_datapath_num5 = "./data/dev2.txt"
        self.dev_datapath_num500 = "./data/dev3.txt"
        # 分类类别路径
        self.class_datapath = "./data/class.txt"

        # 处理后数据路径
        self.process_datapath = "./process_data_result"

        # 停用词路径
        self.stop_words_path = "./data/stopwords.txt"

        # 保存模型路劲
        self.ft_model_save_path = "../model_save_result"