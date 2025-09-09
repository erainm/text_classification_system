#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：app.py
@Create at   ：2025/9/9 09:38
@version     ：V1.0
@Author      ：erainm
@Description : app前端预测
'''
import requests
import time
# 定义预测接口地址
url = 'http://127.0.0.1:8003/predict'

# 构造请求数据
data = {'text': "中华女子学院：本科层次仅1专业招男生"}
# 记录开始时间
start_time = time.time()
# 发送 POST 请求
try:
    response = requests.post(url, json=data)
    # 计算耗时（毫秒）
    elapsed_time = (time.time() - start_time) * 1000
    print(f"请求耗时: {elapsed_time:.2f} ms")

    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        print(f"预测结果: {result['pred_class']}")
    else:
        print(f"请求失败: {response.status_code}, {response.json()['error']}")
except Exception as e:
    print(f"请求出错: {str(e)}")