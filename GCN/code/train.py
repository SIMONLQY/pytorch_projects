import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import re
import pandas as pd
from model import GCNNet
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from GAT import GATNet
import process_data
import predict

def get_train_set(label_list,output,train_id_set,test_id_set,isTrain):
    train_output = torch.stack([ele for i,ele in enumerate(output) if i in train_id_set])
    train_labels = torch.stack([ele for i,ele in enumerate(label_list) if i in train_id_set]).long()
    test_output = torch.stack([ele for i, ele in enumerate(output) if i in test_id_set])
    test_labels = torch.stack([ele for i, ele in enumerate(label_list) if i in test_id_set]).long()
    if isTrain:
        return train_output,train_labels
    else:
        return test_output,test_labels

def test(model,loss_f,feat_Matrix,cites):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        output = model(feat_Matrix, cites)  # 结果为(2708,7),softmax过了
        test_output, test_labels = get_train_set(label_list, output, train_id_set, test_id_set,False)
        test_size = len(test_labels)
        optimizer.zero_grad()
        test_loss += loss_f(test_output, test_labels).item()
        pred = test_output.max(1, keepdim=True)[1]
        correct += pred.eq(test_labels.view_as(pred)).sum().item()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%),Recal:\n'.format(
            test_loss, correct, test_size, 100. * correct / test_size))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    epoch_num = 1000
    gpu = True
    load_flag = False
    model_name = 'gcn'

    # 数据准备
    content_path = "../data/cora.papers"
    cite_path = "../data/cora.cites"
    feat_dim, feat_Matrix, paper_dict, paper_list, label_dict, label_list, cites = \
        process_data.process_data(content_path, cite_path)
    train_id_set = [i for i, ele in enumerate(feat_Matrix) if label_list[i] != -1 and np.random.rand()<0.8]
    test_id_set = [i for i, ele in enumerate(feat_Matrix) if label_list[i] != -1 and np.random.rand()>=0.8]

    # 模型定义
    model = GCNNet(feat_dim, len(label_dict) - 1, len(paper_list))
    if model_name=='gcn':
        model = GCNNet(feat_dim, len(label_dict) - 1, len(paper_list))
    elif model_name == 'gat':
        model = GATNet(feat_dim, len(label_dict) - 1, len(paper_list))
    loss_f = torch.nn.CrossEntropyLoss()
    # 如果使用gpu且为分布式，需要先初始化gpu分布设定，然后将模型，数据转成gpu，数据转出需要转成cpu
    if gpu:
        model = model.cuda()
        loss_f = loss_f.cuda()
    if load_flag:
        # 训练是并行，加载也要并行
        # 但是model=DDP或者model=model.cuda()在整个代码执行过程中不要重复
        # 实际上用torch.nn.DataParallel(model)就可以加载模型了
        model.load_state_dict(torch.load('../trained_model/GCN_model.pth')['model'])

    optimizer = optim.Adam(model.parameters(),lr=0.002)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    loss_list = []
    state = {}

    optimizer.zero_grad()

    # 开始训练
    for epoch in range(epoch_num):
        # 如果使用gpu，数据要转换一下
        if gpu:
            feat_Matrix, cites = feat_Matrix.cuda(), cites.cuda()
            label_list = label_list.cuda()
        output = model(feat_Matrix, cites)  # 结果为(2708,7),softmax过了
        train_output,train_labels = get_train_set(label_list,output,train_id_set,test_id_set,True)
        loss = loss_f(train_output, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if gpu:
            loss = loss.cpu()
        loss_list.append(loss.item())
        if epoch%50==0:
            print("epcoh:{}   loss:{:.6f}".format(epoch + 1, np.mean(loss_list)))
            with open('../log/log.txt', 'w') as f:
                f.write("epcoh:{}   loss:{:.6f} \n".format(epoch + 1, np.mean(loss_list)))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            test(model,loss_f,feat_Matrix,cites)
            model.train()
            if model_name=='gcn':
                torch.save(state, '../trained_model/GCN_model.pth')
            elif model_name=='gat':
                torch.save(state, '../trained_model/GAT_model.pth')
    predict.predict(model)
