#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：fasttext_model_char_default_parameter.py
@Create at   ：2025/9/5 15:52
@version     ：V1.0
@Author      ：erainm
@Description : fasttext模型，使用单字符级别且使用默认参数
'''
# 导入工具包
import fasttext
import datetime
from fasttext_model.config import Config

# 获取当前时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
conf = Config()

# 第一步：模型训练
model = fasttext.train_supervised(input='../process_data_result/train_fasttext_char.txt')
# 第二步：模型保存
model_save_path = conf.ft_model_save_path
model.save_model(model_save_path + f"/fasttext_char_default_{str(current_time)}.bin")
# 第三步：模型预测
# 将输入文本按字符分割
test_text = " ".join(list("《赤壁OL》攻城战诸侯战硝烟又起"))
print(f"test_text --> {test_text}")
print("模型预测 ---> ", model.predict(test_text))

# 第四步：模型词表查看
print(f"查看模型词表前10 ---> {model.words[:10]}")

# 第五步：查看模型子词，上述训练未开启子词，所以这里查到还是词本身
print(f"*模型字词：{model.get_subwords('你')}")

# 第六步：模型测试评估
# 测试集数据路径
test_datapath = "../process_data_result/test_fasttext_char.txt"
res = model.test(test_datapath)
print(res)