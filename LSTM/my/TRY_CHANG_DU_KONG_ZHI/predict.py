import argparse
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import re
import my_dataset
import pandas as pd
from my_model import MyLstm
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import my_word2vec
import my_dataset
embedding, word2id = my_word2vec.get_embed()

def predict(input_path, model):
  input = open(input_path, encoding="utf-8").read().strip()
  # text = np.loadtxt(self.train_path + self.data[index], dtype='str')
  input = my_dataset.tokenize(input)
  input = my_dataset.generate_tensor(input, embedding, word2id)
  input = input.reshape(1,input.shape[0],input.shape[1])
  input_length = [len(s) for s in input]
  input_length = torch.tensor(input_length)
  input = torch.nn.utils.rnn.pad_sequence(input)
  with torch.no_grad():
    input = input.cuda()
    out = model(input,input_length)
    pre = out.max(1,keepdim=True)[1]
    pre = pre.cpu()
    return pre.item()

if __name__ == "__main__":
    text_dir = '../SAPractice/test/'
    files = os.listdir(text_dir)
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    model = MyLstm()
    model = model.cuda()
    model = DDP(model)
    model.load_state_dict(torch.load('./model/my_model.pth')['model'])
    i=0
    with open('./answer.txt', 'a+') as f:
        for file in files:
            pre_value = predict(text_dir+file,model)
            i=i+1
            if i%100==0 :
                print(i)
            f.write(file.split('.')[0] + ' ' + str(pre_value) + '\n')







