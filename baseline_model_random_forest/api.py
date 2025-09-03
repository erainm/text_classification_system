#!/usr/bin/env_set python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：api.py
@Create at   ：2025/9/3 10:21
@version     ：V1.0
@Author      ：erainm
@Description : 模型部署api接口
'''
from flask import Flask, request, jsonify
import pandas as pd
import warnings
from rf_predict_test import predict
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    # 获取JSON输入
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error':"Missing text field in JSON"}), 400

    # 调用预测函数
    result = predict(data)

    # 返回json结果
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)