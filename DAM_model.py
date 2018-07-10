import torch
import torch.nn as nn
import torch.nn.functional as F

class DAM_Sentence(nn.Module):
    def __init__(self, args):
        super(DAM_Sentence, self).__init__()
        self.args = args
        self.hidden_dim = args["lstm_dim"]
        self.layer_num = args["layer_num"]
        self.vec_len = args["vec_len"]
        self.dropout = args["dropout"]
        self.batch_size = args["batch_size"]
        self.remain_l = args["remain_l"]

        self.domain_lstm = nn.LSTM(self.vec_len, self.hidden_dim, bidirectional=True)
        self.domain_fc = nn.Linear(2*self.hidden_dim, 4)
        self.sentiment_lstm = nn.LSTM(self.vec_len, self.hidden_dim, bidirectional=True)
        self.attention = nn.Linear(self.hidden_dim*4, 1)
        self.sentiment_fc = nn.Linear(self.hidden_dim*2, 2)

    def forward(self, x):
        Hd, _ = self.domain_lstm(x.permute(1, 0, 2))
        Hd = Hd.permute(1, 2, 0)
        hd = F.max_pool1d(Hd, Hd.size(2))
        hd = hd.squeeze(2)
        d = self.domain_fc(hd)
        Hs, _ = self.sentiment_lstm(x.permute(1, 0, 2))
        Hs = Hs.permute(1, 0, 2)
        hd = hd.unsqueeze(1)
        Hd = hd.expand(-1, self.remain_l, -1)
        H = torch.cat((Hd, Hs), 2)
        y = F.tanh(self.attention(H))
        alpha = F.softmax(y, dim=1)
        alpha = alpha.permute(0, 2, 1)
        Hs = torch.bmm(alpha, Hs)
        Hs = Hs.squeeze(1)
        output = self.sentiment_fc(Hs)
        return [output, d]


# class DAM_loss(nn.Module):
#     def __init__(self, regular):
#         super(DAM_loss, self).__init__()
#         self.regular = regular
#         self.cross1 = nn.CrossEntropyLoss()
#         self.cross2 = nn.CrossEntropyLoss()
    
#     def forward(self, output, target, d, domain):
#         loss = self.cross1(output, target) + self.regular*self.cross2(d, domain)
#         return loss