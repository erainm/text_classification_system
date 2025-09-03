#!/usr/bin/env_set python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：config.py
@Create at   ：2025/9/3 10:20
@version     ：V1.0
@Author      ：erainm
@Description : 配置文件
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
        self.process_train_datapath = "./process_data/process_train.txt"
        self.process_test_datapath = "./process_data/process_test.txt"
        self.process_dev_datapath = "./process_data/process_dev.txt"
        self.process_dev_num5_datapath = "./process_data/process_dev_num5.txt"
        self.process_dev_num500_datapath = "./process_data/process_dev_num500.txt"

        # 停用词路径
        self.stop_words_path = "./data/stopwords.txt"

        # 保存模型路劲
        self.rf_model_save_path = "./model_save"
        self.model_predict_result = "./result"
        # self.WERKZEUG_RUN_MAIN=True