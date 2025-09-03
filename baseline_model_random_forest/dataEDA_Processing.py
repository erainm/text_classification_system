#!/usr/bin/env_set python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：dataEDA_Processing.py
@Create at   ：2025/9/3 10:22
@version     ：V1.0
@Author      ：erainm
@Description : 数据分析处理
'''
import pandas as pd
import jieba
from config import Config

config = Config()
current_path = config.train_datapath
# current_path = config.dev_datapath
# current_path = config.dev_datapath_num500

# 第一步：读取数据
df = pd.read_csv(current_path, sep="\t", names=["text", "label"])

# 第二步：进行分词处理
def cut_sentence(s):
    """
        对输入文本进行jieba分词，前面了解数据时，已知晓长度大约为30.74，可以设置分词后取30个词并用空格连接
        lcut(s): 返回列表
        :param s: 输入文本
        :return:
    """
    # return ' '.join(list(jieba.cut(s))[:30])
    return ' '.join(jieba.lcut(s))[:30]

df["words"] = df["text"].apply(cut_sentence)

print(df.head(10))

# 第三步：保存处理后的数据

if "train" in current_path:
    df.to_csv(config.process_train_datapath, index=False)
    print(f"train数据已经处理完成，已经成功保存至：{config.process_train_datapath}")
elif "test" in current_path:
    df.to_csv(config.process_test_datapath, index=False)
    print(f"test数据已经处理完成，已经成功保存至：{config.process_test_datapath}")
elif current_path == config.dev_datapath_num5:
    df.to_csv(config.process_dev_num5_datapath, index=False)
    print(f"dev2数据已经处理完成，已经成功保存至：{config.process_dev_num5_datapath}")
elif current_path == config.dev_datapath_num500:
    df.to_csv(config.process_dev_num500_datapath, index=False)
    print(f"dev3数据已经处理完成，已经成功保存至：{config.process_dev_num500_datapath}")
elif current_path == config.dev_datapath:
    df.to_csv(config.process_dev_datapath, index=False)
    print(f"dev数据已经处理完成，已经成功保存至：{config.process_dev_datapath}")