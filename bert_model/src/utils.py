#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：text_classification_system
@File        ：utils.py
@Create at   ：2025/9/9 09:41
@version     ：V1.0
@Author      ：erainm
@Description : 工具类（如：数据处理、构建数据集等）
               utils脚本主要实现了train.txt相关数据读取加载切分，
               以及DataSet、DataLoader的构建，最终输出符合模型需求的数据格式。
'''

import time

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from datetime import timedelta
from tqdm import tqdm

conf = Config()

def load_raw_data(file_path):
    """
        加载原始数据文件，解析为文本和标签
    :param file_path: 数据文件路径
    :return: 包含（文本、标签）的列表
    """
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data"):
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            data.append((text, int(label)))
    print("解析后的文本和标签前5条 ---> \r", data[:5])
    return data

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return x, y

def collate_fn(batch):
    """
    DataLoader的collate_fn,处理分词、统一padding、mask生成和Tensor转换
    :param batch: 批次数据，包含(文本, 标签)
    :tokenizer(BertTokenizer):Bert分词器
    :padding_size(int): 统一padding长度(默认28, 基于文本长度统计)
    device(str): 设备（CPU或者cuda）
    :return: Tuple[torch.Tensor, ...]: (input_ids, seq_len, attention_mask, labels) 的Tensor格式。
    """
    # 提取文本和标签
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 批量分词，自动添加 [CLS] 和 [SEP]  add_special_tokens  # padding，统一处理
    # text_tokens = conf.tokenizer.batch_encode_plus(texts, padding=True)
    # token_ids_list = text_tokens["input_ids"]
    # token_attention_mask_list = text_tokens["attention_mask"]
    # 逐个处理每个文本
    input_ids_list = []
    attention_mask_list = []

    for text in texts:
        # 编码文本
        input_ids = conf.tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            max_length=conf.pad_size,
            truncation=True,
            padding='max_length'
        )

        # 创建attention mask (非pad token为1，pad token为0)
        attention_mask = [1 if token_id != conf.tokenizer.pad_token_id else 0 for token_id in input_ids]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    # 转为 Tensor
    input_ids = torch.tensor(input_ids_list)
    attention_mask = torch.tensor(attention_mask_list)
    labels = torch.tensor(labels)
    print("labels---> \n", labels)
    print("input_ids---> \n", input_ids)
    print("attention_mask---> \n", attention_mask)
    return input_ids, attention_mask, labels

def build_dataloader():
    """
    构建DataLoader,整合数据加载，Dataset、collate_fn
    参数：
        file_path(str) :数据文件路径
        batch_size(int):批次大小
        padding_size(int): 统一padding长度（默认28）
        device(str): 设备(CPU或cuda)
    :return: DataLoader:用于训练的数据集
    """
    # 加载原始数据
    train_data = load_raw_data(conf.train_file)
    test_data = load_raw_data(conf.test_file)
    dev_data = load_raw_data(conf.dev_file)

    # 创建Dataset
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, dev_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        print("input_ids ---> \n", input_ids.tolist())
        print("attention_mask ---> \n", attention_mask.tolist())
        print("labels ---> \n", labels.tolist())