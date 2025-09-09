#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：api.py
@Create at   ：2025/9/9 09:37
@version     ：V1.0
@Author      ：erainm
@Description : 模型部署在线服务
'''
# 模型部署
import fasttext
import jieba
from predict import predict
from flask import Flask, request,jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    # 获取请求数据
    data = request.get_json()
    #预测
    print("-------------预测结果------------")
    result=predict(data)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8003)