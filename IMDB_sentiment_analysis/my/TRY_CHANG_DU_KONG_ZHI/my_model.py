import torch.nn as nn
import torch
import torch.nn.functional as F


class MyLstm(nn.Module):
    def __init__(self,bidrectional):
        super(MyLstm, self).__init__()
        self.hidden_size = 64
        self.embedding_dim = 300
        self.num_layer = 2
        self.bidirectional = bidrectional
        self.bi_num = 2 if self.bidirectional else 1
        self.dropout = 0.1
        # 以上部分为超参数，可以自行修改

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size,
                            self.num_layer, bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * self.bi_num, 20)
        self.fc2 = nn.Linear(20, 2)
        self.sof = nn.Softmax(dim=1)

    def forward(self, x,input_lengths):
        self.device = x.device
        x = x.permute(1,0,2)
        h_0, c_0 = self.init_hidden_state(x.shape[1])
        self.lstm.flatten_parameters()
        # _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        embed_input_x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        encoder_outputs_packed, (h_n, c_n) = self.lstm(embed_input_x_packed, (h_0, c_0))
        # 只要最后一个lstm单元处理的结果，取前向LSTM和后向LSTM的结果进行简单拼接
        if self.bidirectional:
            out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        else:
            out = h_n[-1,:,:]
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(self.device)
        return h_0, c_0

class my_gru(nn.Module):
    pass
