import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

'''
AGG function
'''


class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num

    def forward(self, H, X_node):
        # H = D-1/2 *F*W,还差A
        # H : (N, s) -> (V, s)
        # X_node : (N, )

        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, self.V - 1).float(), 1).to(mask.device)
        mask = (mask == 0).float()

        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)


'''
用于实现GCN的卷积块。
Initialize :
Input :
    in_channel : (int)输入的节点特征维度
    out_channel : (int)输出的节点特征维度
Forward :
Input :
    x : (Tensor)节点的特征矩阵，shape为(N, in_channel)，N为节点个数
    edge_index : (Tensor)边矩阵，shape为(2, E)，E为边个数。
Output :
    out : (Tensor)新的特征矩阵，shape为(N, out_channel)
'''


class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, node_num):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.aggregation = AggrSum(node_num)

    def forward(self, x, edge_index):
        # Add self-connect edges
        edge_index = self.addSelfConnect(edge_index, x.shape[0])

        # Apply linear transform
        x = self.linear(x)

        # Normalize message
        row, col = edge_index

        deg = self.calDegree(row, x.shape[0]).float()
        deg_sqrt = deg.pow(-0.5)  # (N, )
        norm = deg_sqrt[row] * deg_sqrt[col]

        # Node feature matrix
        tar_matrix = torch.index_select(x, dim=0, index=col)
        tar_matrix = norm.view(-1, 1) * tar_matrix  # (E, out_channel)
        # tar_matrix 是13566，16尺寸，是按照edge序号顺序抽取排列的feature_matrix
        # Aggregate information
        aggr = self.aggregation(tar_matrix, row)  # (N, out_channel)
        return aggr

    def calDegree(self, edges, num_nodes):
        ind, deg = np.unique(edges.cpu().numpy(), return_counts=True)
        deg_tensor = torch.zeros((num_nodes,), dtype=torch.long)
        deg_tensor[ind] = torch.from_numpy(deg)
        return deg_tensor.to(edges.device)

    def addSelfConnect(self, edge_index, num_nodes):
        selfconn = torch.stack([torch.range(0, num_nodes - 1, dtype=torch.long)] * 2,
                               dim=0).to(edge_index.device)
        return torch.cat(tensors=[edge_index, selfconn],
                         dim=1)


'''
构建模型，使用两层GCN，第一层GCN使得节点特征矩阵
    (N, in_channel) -> (N, out_channel)
第二层GCN直接输出
    (N, out_channel) -> (N, num_class)
激活函数使用relu函数，网络最后对节点的各个类别score使用softmax归一化。
'''


class GCNNet(nn.Module):
    def __init__(self, feat_dim, num_class, num_node):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16, num_node)
        self.conv2 = GCNConv(16, num_class, num_node)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
