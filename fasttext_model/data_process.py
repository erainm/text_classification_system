#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：data_process.py
@Create at   ：2025/9/5 15:14
@version     ：V1.0
@Author      ：erainm
@Description : 数据处理
'''
import os.path

"""
    分析：FastText格式：
        假设原始样本：我喜欢文本分类
        单字符级别处理： __label__类别 我 喜 欢 文 本 分 类
        词级别分词处理：__label_类别 我 喜欢 文本 分类
"""

import jieba
from config import *
conf=Config()

# 第一步：加载原始数据
class_datapath = conf.class_datapath
# datapath = conf.train_datapath
datapath = conf.dev_datapath
# datapath = conf.test_datapath

if not os.path.exists(class_datapath) or not os.path.exists(datapath):
    print("类别文件及原始数据路径：", class_datapath, datapath)
    print("文件不存在，请检查文件是否存在！")
    # exit(1)  # 添加这行来终止程序执行

# 第二步：设置预处理后文件保存路径(确认使用那种处理方式：单字符还是词级别)
use_char_segmentation = True
# use_char_segmentation = False
if use_char_segmentation:
    # 单字符级文件写入路径
    if 'train' in datapath:
        output_file = conf.process_datapath + '/train_fasttext_char.txt'
    elif 'test' in datapath:
        output_file = conf.process_datapath + '/test_fasttext_char.txt'
    else:
        output_file = conf.process_datapath + '/dev_fasttext_char.txt'
else:
    # 词级别文件写入路径
    if 'train' in datapath:
        output_file = conf.process_datapath + '/train_fasttext_jieba.txt'
    elif 'test' in datapath:
        output_file = conf.process_datapath + '/test_fasttext_jieba.txt'
    else:
        output_file = conf.process_datapath + '/dev_fasttext_jieba.txt'

# 第三步：ID与类别映射（读取类别class.txt，生成ID到类别名的映射字典）
id2name = {}
with open(class_datapath, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        id2name[idx] = line.strip() # 去除换行符，映射索引到类别名

print("ID与类别映射 ---> ", id2name)

# 第四步：FastText训练数据构造
datas = []
with open(datapath, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip() # 去除换行和空白
        if not line:
            continue
        text, label = line.split('\t') # 以制表符分割
        label_name = f"__label__{id2name[int(label)]}" # 转换标签
        text = text.replace('：','')
        # 字符级别分词
        words = list(text) if use_char_segmentation else  jieba.cut(text)
        text_processed = ' '.join(word for word in words if word.strip())
        fasttext_line = f"{label_name} {text_processed}"
        datas.append(fasttext_line)

# 第五步：保存预处理后的数据
with open(output_file, 'w', encoding='utf-8') as f:
    for line in datas:
        f.write(line + '\n')

print("前5行数据", datas[:5])
print(f"数据已经保存到{output_file}")
