#!/usr/bin/env_set python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：app.py
@Create at   ：2025/9/3 10:21
@version     ：V1.0
@Author      ：erainm
@Description : 前端预测界面
'''
import streamlit as st
import requests
import time

# 设置页面标题
st.title("文本分类预测")

# 创建输入框
text_input = st.text_area("请输入要预测的文本：", "中国人民公安大学2012年硕士研究生目录及书目")

# 创建预测按钮
if st.button("预测"):
    # 构造请求数据
    data = {'text':text_input}
    url = "http://127.0.0.1:8001/predict"
    # 记录开始时间
    start_time = time.time()

    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        # 计算耗时
        elapsed_time = (time.time() - start_time) * 1000

        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            st.success(f"预测结果为:{result["pred_class"]}")
            st.info(f"请求耗时：{elapsed_time:.2f} ms")
        else:
            st.error(f"请求失败：{response.status_code}, {response.json()["error"]}")
    except requests.exceptions.ConnectionError:
        st.error("连接错误：无法连接到预测服务，请确保Flask API服务正在运行")
    except requests.exceptions.Timeout:
        st.error("请求超时：请检查网络连接")
    except Exception as e:
        st.error(f"请求出错: {str(e)}")

# 运行提示
st.write("请确保 Flask API 服务已在 localhost:8001 运行")