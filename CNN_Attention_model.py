import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Attention_Sentence(nn.Module):
    
    def __init__(self, args):
        super(CNN_Attention_Sentence, self).__init__()
        self.args = args

        self.vec_len = args["vec_len"]
        self.filter_num = args["filter_num"]
        self.kernel_sizes = args["kernel_sizes"]
        self.kernel_num = args["kernel_num"]
        self.dropout = args["dropout"]
        self.sentence_l = args["remain_l"]
        
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.vec_len)) for k in self.kernel_sizes])
        self.attentions = nn.ModuleList([nn.Linear(self.sentence_l-k+1, self.sentence_l-k+1) for k in self.kernel_sizes])
        self.att_weights = nn.ParameterList([nn.Parameter(torch.rand(self.sentence_l-k+1, 1)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear((self.sentence_l+1)*len(self.kernel_sizes)-sum(self.kernel_sizes), 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        Xs = []
        for i in range(self.filter_num):
            h = F.relu(self.convs[i](x))
            h = h.squeeze(3)
            u = self.attentions[i](h)
            u = F.tanh(u)
            u = torch.matmul(u, self.att_weights[i])
            u = F.softmax(u, dim=1)
            u = u.permute(0, 2 ,1)
            output = torch.bmm(u, h)
            output = output.squeeze(1)
            Xs.append(output)
        x = torch.cat(Xs, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
