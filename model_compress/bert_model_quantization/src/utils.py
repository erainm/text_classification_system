#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：utils.py
@Create at   ：2025/9/11 15:23
@version     ：V1.0
@Author      ：erainm
@Description : 主要实现了训练集相关数据读取加载切分，以及DataSet、DataLoader的构建，最终输出符合模型需求的数据格式
'''
import torch
from torch.utils.data import Dataset, DataLoader

from model_compress.bert_model_quantization.src.config import Config
from tqdm import tqdm
import time

conf = Config()

"""
    数据处理：
    中华女子学院：本科层次仅1专业招男生  3
    两天价网站背后重要迷雾：做个网站究竟要多少钱  4
    
    处理为：
        [('中华女子学院：本科层次仅1专业招男生', 3), ('两天价网站背后重要迷雾：做个网站究竟要多少钱', 4)]
"""
def load_raw_data(file_path):
    """
    读取原始数据文件，解析为文本和标签。
    参数：
        file_path (str): 数据文件路径（如dev2.txt）。
    返回：
        List[Tuple[str, int]]: 包含(文本, 标签)的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            data.append((text, int(label)))
    print("处理后的前五行数据 ---> ", data[:5])
    return data

# 构建自定义数据集
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return x, y

# 构建collate_fn函数，作用于DataLoader
def collate_fn(batch):
    """
    collate_fn是dataloader为了解决进入模型训练的数据不符合要求的进一步处理，例如batch级别数据处理长度、数据数值化等
    :param batch: 批次数据，包含（文本、标签）
    :tokenizer:Bert分词器
    :padding_size 统一Padding长度(默认28，基于文本长度统计)
    :return: Tuple[torch.Tensor, ...]: (input_ids, seq_len, attention_mask, labels) 的Tensor格式。
    """
    # 提取文本和标签
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # 批量分词，自动添加[CLS] 和 [SEP]
    text_tokens = conf.tokenizer.batch_encode_plus(texts, padding=True)
    token_ids_list = text_tokens["input_ids"]
    token_attention_mask_list = text_tokens["attention_mask"]
    # 转为Tensor
    input_ids = torch.tensor(token_ids_list)
    attention_mask = torch.tensor(token_attention_mask_list)
    labels = torch.tensor(labels)

    return input_ids, attention_mask, labels


# 构建Dataloader
def build_dataloader():
    """
    构建dataloader，整合数据加载、Dataset和collate_fn
    :param
        file_path: 数据文件
        batch_size: 批次大小
        padding_size: 同一padding长度（默认28）
        device: 设备(cpu或cuda)
    :return: DataLoader：用于训练的DataLoader
    """
    # 加载原始数据
    train_data = load_raw_data(conf.train_file)
    test_data = load_raw_data(conf.test_file)
    dev_data = load_raw_data(conf.dev_file)
    # 创建DataSet
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, dev_dataloader