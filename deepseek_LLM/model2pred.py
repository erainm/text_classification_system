#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：model2pred.py
@Create at   ：2025/9/9 16:55
@version     ：V1.0
@Author      ：erainm
@Description : 模型预测
'''
import time
import os
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

def model2pred(Text):
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("base_url"))

    system_prompt = '''
    你是一个优秀的文本分类师，能把给定的用户query划分到正确的类目中。现在请你根据给定信息和要求，为给定用户query，从备选类目中选择最合适的类目。

    下面是“参考案例”即被标注的正确结果，可供参考：
    文本：中国国家乒乓球队击败日本
    类别：sports

    备选类目：
    finance,realty,stocks,education,science,society,politics,sports,game,entertainment

    请注意：
    1. 用户query所选类目，仅能在【备选类目】中进行选择，用户query仅属于一个类目。
    2. “参考案例”中的内容可供推理分析，可以仿照案例来分析用户query的所选类目。
    3. 请仔细比对【备选类目】的概念和用户query的差异。
    4. 如果用户query也不属于【备选类目】中给定的类目，或者比较模糊，请选择“拒识”。
    5. 请在“文本类别：”后回复结果，不需要说明理由。

    类别:
    '''
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": Text},
            ],
            stream=False
        )

        result = response.choices[0].message.content
        elapsed_time = (time.time() - start_time) * 1000

        return result, elapsed_time

    except APIStatusError as e:
        if e.status_code == 402:
            error_msg = "API余额不足，请充值后再试"
        else:
            error_msg = f"API状态错误: {e.message}"
        print(f"错误：{error_msg}")
        return error_msg, 0
    except APIConnectionError:
        error_msg = "网络连接错误，请检查网络设置"
        print(f"错误：{error_msg}")
        return error_msg, 0
    except RateLimitError:
        error_msg = "请求频率过高，请稍后再试"
        print(f"错误：{error_msg}")
        return error_msg, 0
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        print(f"错误：{error_msg}")
        return error_msg, 0


if __name__ == '__main__':
    result, elapsed_time = model2pred("今日大A净流入520亿，全市超3500家上涨。")
    print(f'*预测类别：{result}')
    print(f'*请求耗时：{elapsed_time:.2f}ms')