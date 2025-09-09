#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：config.py
@Create at   ：2025/9/9 16:54
@version     ：V1.0
@Author      ：erainm
@Description : 配置文件
'''
class Config(object):
    def __init__(self):
        # 原始数据路径
        self.train_datapath = "../data/train.txt"
        self.test_datapath = "../data/test.txt"
        self.dev_datapath = "../data/dev.txt"
        self.class_datapath = "../data/class.txt"


if __name__ == '__main__':
    conf = Config()
    print(conf.train_datapath)
    print(conf.test_datapath)