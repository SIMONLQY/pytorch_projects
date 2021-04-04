from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import os
import pandas as pd

glove_file = '../glove.42B.300d.txt'
tmp_file = 'test_word2vec.txt'
glove2word2vec(glove_file, tmp_file)
Word2VecModel = KeyedVectors.load_word2vec_format(tmp_file)
print(Word2VecModel.wv['my'].shape)

"""得到需要的word-embedded模型而不是全部"""
ori_path = '../SAPractice/train/'
files = os.listdir(ori_path)
word2vec_dir = "test_word2vec.txt"
print("read wv file")
wv = KeyedVectors.load_word2vec_format(word2vec_dir, binary=False)
vocabulary = []
print("start word2vec load ......")
for file in files:
    text = open(ori_path + file, encoding="utf-8").read().strip()
    text = main.tokenize(text)
    vocabulary.extend(text)

vocabulary = list(set(vocabulary))
vocabulary_vector = {}
k = 0
for word in vocabulary:
    if word in wv:
        vocabulary_vector[word] = wv[word]
    else:  # UNK
        k = k + 1
        print(k)
        vocabulary_vector[word] = [0 for i in range(300)]
# 储存词汇-向量字典，由于json文件不能很好的保存numpy词向量，故使用csv保存
pd.DataFrame(vocabulary_vector).to_csv("my_vocabulary_vector.csv")