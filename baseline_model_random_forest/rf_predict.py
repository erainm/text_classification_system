#!/usr/bin/env_set python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：rf_predict.py
@Create at   ：2025/9/3 10:22
@version     ：V1.0
@Author      ：erainm
@Description : 随机森林模型预测
'''
import pandas as pd
import pickle
from config import Config
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, recall_score
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
# 加载配置
conf = Config()

# 第一步：加载保存的模型和向量化器
print("第一步：加载模型和向量化器...")
with open(conf.rf_model_save_path + '/rf_model_.pkl', 'rb') as f:
    model = pickle.load(f)
with open(conf.rf_model_save_path + '/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# 第二步：读取dev数据
print("第二步：读取dev数据...")
dev_df = pd.read_csv(conf.process_dev_datapath)
print(">>> dev数据前5行:")
print(dev_df.head(5))

# 第三步：通过tfidf向量器，转换为数值特征
print("第三步：转换dev数据为数值...")
dev_features = tfidf.transform(dev_df['words'])

# 第四步：进行预测及保存
print("第四步：进行预测...")
dev_predictions = model.predict(dev_features)

# 第五步：保存预测结果
print("第五步：保存预测结果...")
output_df = pd.DataFrame({'words': dev_df['words'], 'predicted_label': dev_predictions})
output_path = conf.model_predict_result + '/dev_predictions.csv'
output_df.to_csv(output_path, index=False)
print(f"预测结果已保存到 {output_path}")
print(f"预测结果前5行:\n{output_df.head(5)}" )

# 第六步：模型评估
print("第六步：模型评估...")
label = dev_df["label"]
print(f">>> 准确率Accuracy：{accuracy_score(label, dev_predictions)}")
print(f">>> 精确率Precision：{precision_score(label, dev_predictions, average='micro')}")
print(f">>> 召回率Recall：{recall_score(label, dev_predictions, average='micro')}")

report = classification_report(label, dev_predictions)
print(f">>> 混淆矩阵:\n{confusion_matrix(label, dev_predictions)}")
print(f">>> 分类报告:\n{report}")