#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：know_data_info.py
@Create at   ：2025/9/3 15:45
@version     ：V1.0
@Author      ：erainm
@Description : 了解训练集数据信息
'''
from collections import Counter
import pandas as pd
from config import Config

# 初始化配置文件
conf = Config()

# 读取数据

# 设置读取的文件路径进行处理及分析
current_path = conf.dev_datapath

# 第一步：读取文件数据(指定文件的列明text和label),并了解数据基本信息
data = pd.read_csv(current_path, sep="\t", names=["text", "label"])
print(f"查看数据前5行数据 --->\n {data.head(5)}")
print(f"\n查看数据总数据量 ---> \n {len(data)}")
# 第二步：统计标签分布
label_counts = Counter(data["label"])
print(f"\n 查看标签分布 --->")
for label, count in label_counts.items():
    print(f"标签{label}: {count}次")

# 第三步：计算标签比例
total_rows = len(data)
print("\n *******************标签比例*******************")
for label, count in label_counts.items():
    precent = (count / total_rows) * 100
    print(f"标签{label}: {precent:.2f}%")

# 第四步：文本长度分析
data["text_length"] = data["text"].str.len() # 计算每条文本的字符数
print(f"\n 文本长度前十行 ---> \n  {data[["text", "text_length"]].head(10)}")
print("\n *******************文本长度统计*******************")
print(f"平均长度μ：{data["text_length"].mean():.2f} 字符")
print(f"标准差长度δ：{data["text_length"].std():.2f} 字符")
print(f"最大长度：{data["text_length"].max():.2f} 字符")
print(f"最小长度：{data["text_length"].min():.2f} 字符")
use_len = data["text_length"].mean() + (3 * data["text_length"].std())
print(f"一般取长度可遵循：len = μ * 3δ原则，则可取长度为：{use_len:.2f}")