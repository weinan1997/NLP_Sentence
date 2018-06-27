import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Attention_Sentence(nn.Module):
    
    def __init__(self, args):
        super(CNN_Attention_Sentence, self).__init__()
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
        self.fc = nn.Linear(self.sentence_l, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        Xs = []
        for i in range(self.filter_num):
            h = F.pad(x, (0, 0, 0, self.kernel_sizes[i]-1))
            h = F.relu(self.convs[i](h))
            h = h.squeeze(3)
            Xs.append(h)
        x = torch.cat(Xs, 1)
        u = F.tanh(self.attention(x))
        u = F.softmax(torch.matmul(u, self.att_weight), dim=1)
        u = u.permute(0, 2, 1)
        x = torch.bmm(u, x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
