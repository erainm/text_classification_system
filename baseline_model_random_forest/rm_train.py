#!/usr/bin/env_set python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：rm_train.py
@Create at   ：2025/9/3 10:23
@version     ：V1.0
@Author      ：erainm
@Description : 随机森林模型训练
'''
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from tqdm import tqdm  # 引入 tqdm 用于进度条
from config import Config

pd.set_option("display.expand_frame_repr", False) # 避免宽表格换行
pd.set_option("display.max_columns", None) # 确保所有列可见

conf = Config()

# 第一步：读取训练集数据
df = pd.read_csv(conf.process_train_datapath)
words = df["words"]
labels = df["label"]
print(df.head(5))

# 第二步：将文本转化为数值特征
# 读取停用词文件
stop_words = open(conf.stop_words_path, 'r', encoding='utf-8').read().split()
transfer = TfidfVectorizer(stop_words=stop_words)
words_features = transfer.fit_transform(words)
# 2.1 查看特征长度
print(f"特征：{words_features}")
# 2.2 特征维度 形状
print(f"特征维度形状为：{words_features.shape}")
# 2.3 特征名字
print(f"特征名字为：{transfer.get_feature_names_out()}")
# 2.4 特征词表
print(f"特征词表vocab为：{transfer.vocabulary_}")

# 第三步： 划分训练集和测试集，进行模型训练和模型评估
# 3.1 划分数据集
x_train, x_test, y_train, y_test = train_test_split(words_features, labels, train_size=0.8, random_state=22)
# 3.2 使用随机森林模型进行训练
model = RandomForestClassifier(verbose=1) # 设置 verbose=1 以输出训练进度
print("随机森林模型正在训练中……")
# 3.3 使用 tqdm 包装 model.fit 来显示进度条
for _ in tqdm(range(1), desc="RandomForest模型训练进度...."):
    model.fit(x_train, y_train)

# 3.4 模型预测并评估
print("模型预测评估 ---> ")
y_pred = model.predict(x_test)
print(">>> 预测结果为：", y_pred)
print(">>> 预测准确率为：", accuracy_score(y_test, y_pred))
print(">>> 预测精确率(micro)为：", precision_score(y_test, y_pred, average="micro"))
print(">>> 预测召回率(micro)为：", recall_score(y_test, y_pred, average="micro"))
print(">>> 预测F1 Score(micro)为：", f1_score(y_test, y_pred, average="micro"))

# 第四步：保存模型和向量化器
print("保存模型和向量化器 ---> ")
with open(conf.rf_model_save_path + "/rf_model_.pkl", "wb") as f:
    pickle.dump(model, f)
with open(conf.rf_model_save_path + "/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(transfer, f)

print("模型和向量化器，保存成功！")