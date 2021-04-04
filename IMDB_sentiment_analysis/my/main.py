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
from torch.optim import lr_scheduler

def test(model,test_loader,loss_f,test_size):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader, 0):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += loss_f(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= (i + 1)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%),Recal:\n'.format(
            test_loss, correct, test_size, 100. * correct / test_size))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    epoch_num = 100
    gpu = True
    batch_size = 64
    sentence_max_size = 500
    load_flag = True
    bidrectional = True

    # 获取embedding模型
    print("Embedding Model preparing...")
    embedding, word2id = my_word2vec.get_embed()
    print("Embedding Model Got")

    # 数据准备
    full_dataset = my_dataset.MyDataset('../SAPractice/train/', sentence_max_size, embedding, word2id)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # 模型定义
    model = MyLstm(bidrectional=bidrectional)
    loss_f = torch.nn.CrossEntropyLoss()
    # 如果使用gpu且为分布式，需要先初始化gpu分布设定，然后将模型，数据转成gpu，数据转出需要转成cpu
    if gpu:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23457', rank=0, world_size=1)
        model = model.cuda()
        model = DDP(model)
        loss_f = loss_f.cuda()
    if load_flag:
        # 训练是并行，加载也要并行
        # 但是model=DDP或者model=model.cuda()在整个代码执行过程中不要重复
        # 实际上用torch.nn.DataParallel(model)就可以加载模型了
        model.load_state_dict(torch.load('./model/my_model.pth')['model'])

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_list = []
    state = {}

    # 开始训练
    for epoch in range(epoch_num):
        for i, (x, y) in enumerate(train_loader):
            # 如果使用gpu，数据要转换一下
            if gpu:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            # print(x.shape)
            # assert 0==2
            output = model(x)
            loss = loss_f(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 如果使用gpu，需要转换一下取出loss
            if gpu:
                loss = loss.cpu()
            loss_list.append(loss.item())
            if i % 100 == 0:
                print("epcoh:{}  iteration:{}   loss:{:.6f}".format(epoch + 1, i, np.mean(loss_list)))
                with open('./log/log.txt', 'a+') as f:
                    f.write("epcoh:{}  idx:{}   loss:{:.6f} \n".format(epoch + 1, i, np.mean(loss_list)))
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        test(model,test_loader,loss_f,test_size)
        model.train()
        mode = ('bi' if bidrectional else 'single')
        # torch.save(state, './model/my_model_'+ mode + '.pth')
        torch.save(state, './model/my_model.pth')
