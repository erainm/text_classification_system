#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：fasttext_mode_word_default_parameter.py
@Create at   ：2025/9/5 17:08
@version     ：V1.0
@Author      ：erainm
@Description : FastText词级别模型，使用默认参数
'''
# 导入工具包
import fasttext
from fasttext_model.config import Config
import datetime

#获取时间
current_time=datetime.datetime.now().date().today().strftime("%Y%m%d")
# 配置文件实例化
conf=Config()

# 第一步：训练模型
model = fasttext.train_supervised(input = "../process_data_result/train_fasttext_jieba.txt")

# 第二步：保存模型
save_model_path = conf.ft_model_save_path + f"/fasttext_jieba_default_{str(current_time)}.bin"
model.save_model(save_model_path)

# 第三步：模型预测
print(model.predict("名师 详解 考研 复试 英语听力 备考 策略"))

# 第四步：模型评估
res = model.test("../process_data_result/test_fasttext_jieba.txt")
print(f"测试结果: 样本数={res[0]}, 精确率={res[1]:.4f}, 召回率={res[2]:.4f}")