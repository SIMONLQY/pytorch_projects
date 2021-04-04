import numpy as np
import pandas as pd
import torch.nn as nn
import torch


def get_embed():
    # 根据已经训练好的词向量模型，生成Embedding对象
    vocabulary_vector = dict(pd.read_csv("my_vocabulary_vector.csv"))
    # 此时需要将字典中的词向量np.array型数据还原为原始类型，方便以后使用
    voca_num = []
    word2id = {}  # word2id是一个字典，存储{word：id}的映射
    i = 0
    for key, value in vocabulary_vector.items():
        voca_num.append(np.array(value))
        word2id[key] = i
        i += 1
    print("词表长度：" + str(len(voca_num)))
    print("vocabulary vector load succeed")
    voca_num = np.array(voca_num)

    # # 根据已经训练好的词向量模型，生成Embedding对象
    # embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

    embedding = nn.Embedding(len(vocabulary_vector), 300)
    embedding.weight.data.copy_(torch.from_numpy(voca_num))
    embedding.weight.requires_grad = False
    # print(embedding.embedding_dim)
    # print(embedding.weight[word2id['my']])
    return embedding, word2id
