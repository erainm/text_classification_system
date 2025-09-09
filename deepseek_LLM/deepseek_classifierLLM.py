#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：deepseek_classifierLLM.py
@Create at   ：2025/9/9 16:52
@version     ：V1.0
@Author      ：erainm
@Description : deepseek大模型完成文本分类
'''
from openai import OpenAI    # pip install openai
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("base_url"))

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是谁"},
    ],
    stream=False
)

print(response.choices[0].message.content)