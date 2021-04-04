import os
import re
import argparse
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from my_model import MyLstm
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

def generate_tensor(sentence, embedding, word2id):
    """
    对一篇评论生成对应的词向量矩阵
    :param sentence:一篇评论的分词列表
    :param sentence_max_size:认为设定的一篇评论的最大分词数量
    :param embedding:词向量对象
    :param word2id:字典{word:id}
    :return:一篇评论的词向量矩阵
    """
    tensor = torch.zeros([len(sentence), embedding.embedding_dim])
    # UNK和PAD都取0
    for index in range(0, len(sentence)):
        word = sentence[index]
        if word in word2id:
            vector = embedding.weight[word2id[word]]
            tensor[index] = vector
        elif word.lower() in word2id:
            vector = embedding.weight[word2id[word.lower()]]
            tensor[index] = vector

    return tensor  # tensor是二维的，必须扩充为三维，否则会报错
# 分词的API
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]


class MyDataset(Dataset):
    def __init__(self, train_path, embedding, word2id, is_train=True):
        ## record_path:记录数据路径及对应label的文件
        self.is_train = is_train
        self.embedding = embedding
        self.word2id = word2id
        self.train_path = train_path
        self.data = os.listdir(train_path)

    # 获取单条数据
    def __getitem__(self, index):
        text = open(self.train_path + self.data[index], encoding="utf-8").read().strip()
        # text = np.loadtxt(self.train_path + self.data[index], dtype='str')
        text = tokenize(text)
        tensor = generate_tensor(text, self.embedding, self.word2id)
        label = int(1 if self.data[index].split('_')[1].split('.')[0] == 'pos' else 0)
        return tensor, label

    # 数据集长度
    def __len__(self):
        return len(self.data)