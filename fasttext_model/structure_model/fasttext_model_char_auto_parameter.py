#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：fasttext_model_char_auto_parameter.py
@Create at   ：2025/9/5 16:39
@version     ：V1.0
@Author      ：erainm
@Description : 
'''
"""
开启auto模式的模型训练：
    autotuneValidationFile参数需要指定验证数据集所在的路径它将在验证集是使用随机搜索的方法寻找最优的超参数
    使用autotuneDuration参数可以控制随机搜索的时间，单次搜索 默认是300，单位秒.根据不同的需求, 可以延长或者缩短时间.
    自动调节的超参数包含这些内容:
    lr                         学习率 default 0.1
    dim                        词向量维度 default 100
    ws                         上下文窗口大小 default 5， cbow
    epoch                      epochs 数量 default 5
    minCount                   最低词频 default 5
    wordNgrams                 n-gram设置 default 1
    loss                       损失函数 {hs,softmax} default softmax
    minn                       最小字符长度 default 0
    maxn                       最大字符长度 default 0
    dsub                       dsub ，全称为 "dimension subsampling"，用于控制输入向量的子采样率，以减少模                                                      型的计算复杂度和内存占用。它通过降低词嵌入（输入向量）的维度来加速训练和压缩模型。
    如何观察超参数调整过程，verbose: 该参数决定日志打印级别, 当设置为3, 可以将当前正在尝试的超参数打印出来。
"""
import fasttext
from fasttext_model.config import Config
import datetime

# 获取当前日期
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
# 第一步：导入配置文件
conf = Config()

# 第二步：模型训练
model = fasttext.train_supervised(
    input = '../process_data_result/train_fasttext_char.txt',
    autotuneValidationFile = '../process_data_result/dev_fasttext_char.txt',
    autotuneDuration = 300,
    thread=1,  # 单线程，确保可复现性
    verbose=3  # 输出调参过程
)

# 第三步：模型保存
model_save_path = conf.ft_model_save_path
model.save_model(model_save_path + f"/fasttext_char_auto_{str(current_time)}.bin")
# 第四步：模型预测
# 将输入文本按字符分割
test_text = " ".join(list("俄达吉斯坦共和国一名区长被枪杀"))
print(f"test_text --> {test_text}")
pred_label, pred_prob = model.predict(test_text)
print(f"预测结果: 标签={pred_label[0]}, 概率={pred_prob[0]:.4f}")

# 第五步：查看模型子词，上述训练未开启子词，所以这里查到还是词本身
word = "好"
subwords, subword_ids = model.get_subwords(word)
print(f"*词'{word}'的子词:{subwords}")
print(f"*子词ID:{subword_ids}")

# 第六步：模型测试评估
res = model.test('../process_data_result/test_fasttext_char.txt')
print(f"测试结果: 样本数={res[0]}, 精确率={res[1]:.4f}, 召回率={res[2]:.4f}")