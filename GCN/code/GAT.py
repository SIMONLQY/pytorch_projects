import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F


class GATNet(torch.nn.Module):
    def __init__(self,in_channel, out_channel, node_num):
        super(GATNet,self).__init__()
        self.gat1=GATConv(in_channel,8,8,dropout=0.6)
        self.gat2=GATConv(64,out_channel,1,dropout=0.6)

    def forward(self,x, edge_index):
        x=self.gat1(x,edge_index)
        x=self.gat2(x,edge_index)
        return x
