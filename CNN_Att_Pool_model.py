import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Att_Pool_Sentence(nn.Module):
    
    def __init__(self, args):
        super(CNN_Att_Pool_Sentence, self).__init__()
        self.args = args

        self.vec_len = args["vec_len"]
        self.kernel_sizes = args["kernel_sizes"]
        self.filter_num = len(self.kernel_sizes)
        self.kernel_num = args["kernel_num"]
        self.dropout = args["dropout"]
        self.sentence_l = args["remain_l"]

        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.vec_len)) for k in self.kernel_sizes])
        self.attention = nn.Linear(self.sentence_l, self.sentence_l)
        self.att_weight = nn.Parameter(torch.rand(self.sentence_l, 1))
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.sentence_l+self.filter_num*self.kernel_num, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        Xs = []
        Ms = []
        for i in range(self.filter_num):
            h = F.pad(x, (0, 0, 0, self.kernel_sizes[i]-1))
            h = F.relu(self.convs[i](h))
            h = h.squeeze(3)
            m = F.max_pool1d(h, h.size(2))
            m = m.squeeze(2)
            Xs.append(h)
            Ms.append(m)
        x = torch.cat(Xs, 1)
        m = torch.cat(Ms, 1)
        u = F.tanh(self.attention(x))
        u = F.softmax(torch.matmul(u, self.att_weight), dim=1)
        u = u.permute(0, 2, 1)
        x = torch.bmm(u, x)
        x = x.squeeze(1)
        x = torch.cat((x, m), 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x