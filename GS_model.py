import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GS_Sentence(nn.Module):
    def __init__(self, args):
        super(GS_Sentence, self).__init__()
        self.args = args
        self.domain_num = 4
        self.hidden_dim = args["hidden_dim"]

        self.general = General(args)
        self.specifics = [Specific(args, k) for k in range(self.domain_num)]
        if torch.cuda.is_available():
            self.general = self.general.cuda(args["GPU"])
            self.specifics = [self.specifics[k].cuda(args["GPU"]) for k in range(self.domain_num)]
        self.fc = nn.Linear(self.hidden_dim*2*2, 2)   

    def forward(self, x, z):
        general_output = self.general(x)
        specific_outputs = [self.specifics[k](x) for k in range(self.domain_num)]
        all_outputs = []
        for row in range(x.size(0)):
            all_outputs.append(specific_outputs[z[row]][row])
        all_outputs = torch.cat(all_outputs, 0)
        all_outputs = all_outputs.unsqueeze(1)
        all_outputs = torch.cat((all_outputs, general_output), 2)
        all_outputs.squeeze(2)
        final_output = self.fc(all_outputs)
        final_output = final_output.squeeze(1)
        general_output = general_output.squeeze(1)
        specific_outputs = [specific_outputs[k].squeeze(1) for k in range(self.domain_num)]
        return final_output, general_output, specific_outputs

    def loss(self, final_output, general_output, specific_outputs, y, z, lambda1, lambda2):
        loss = F.cross_entropy(final_output, y)
        loss = loss + lambda1 * self.general.loss(general_output, y)
        for i in range(self.domain_num):
            loss = loss + lambda2 * self.specifics[i].loss(specific_outputs[i], y, z)
        return loss



class General(nn.Module):
    def __init__(self, args):
        super(General, self).__init__()
        self.args = args
        
        self.hidden_dim = args["attention_dim"]
        self.vec_len = args["vec_len"]
        self.lstm = nn.LSTM(self.vec_len, self.hidden_dim, bidirectional=True)
        self.att_weight = nn.Parameter(torch.rand(2*self.hidden_dim, 1))
        self.attention = nn.Linear(self.hidden_dim*2, 2*self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim*2, 2)

    def forward(self, x):
        h, _ = self.lstm(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        u = self.attention(h)
        u = F.tanh(u)
        u = torch.matmul(u, self.att_weight)
        u = F.softmax(u, dim=1)
        u = u.permute(0, 2, 1)
        output = torch.bmm(u, h)
        return output
    
    def loss(self, x, y):
        output = self.fc(x)
        loss = F.cross_entropy(output, y)
        return loss


class Specific(nn.Module):
    def __init__(self, args, domain):
        super(Specific, self).__init__()
        self.args = args
        self.domain = domain
        self.hidden_dim = args["attention_dim"]
        self.vec_len = args["vec_len"]
        self.lstm = nn.LSTM(self.vec_len, self.hidden_dim, bidirectional=True)
        self.att_weight = nn.Parameter(torch.rand(2*self.hidden_dim, 1))
        self.attention = nn.Linear(self.hidden_dim*2, 2*self.hidden_dim)
        self.fc = nn.Linear(2*self.hidden_dim, 3)


    def forward(self, x):
        h, _ = self.lstm(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        u = self.attention(h)
        u = F.tanh(u)
        u = torch.matmul(u, self.att_weight)
        u = F.softmax(u, dim=1)
        u = u.permute(0, 2, 1)
        output = torch.bmm(u, h)
        return output

    def loss(self, x, y, z):
        y = y.clone()
        for i in range(y.shape[0]):
            if z[i] != self.domain:
                y[i] = 2
        output = self.fc(x)
        loss = F.cross_entropy(output, y)
        return loss

            
        