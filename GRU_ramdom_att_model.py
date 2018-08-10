import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_random_att_Sentence(nn.Module):

    def __init__(self, args):
        super(GRU_random_att_Sentence, self).__init__()
        self.args = args
        
        self.hidden_dim = args["attention_dim"]
        self.vec_len = args["vec_len"]
        self.batch_size = args["batch_size"]
        self.word2vec = args["W"]
        self.dp = args["dropout"]
        self.domain_num = args["domain_num"]
        self.seq_len = args["remain_l"]
        self.gpu = args["GPU"]
        self.att_len = 500

        self.embedding = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.gru = nn.GRU(self.vec_len, self.hidden_dim, bidirectional=True)
        self.att_weight = nn.Parameter(torch.FloatTensor(self.att_len, 1))
        torch.nn.init.xavier_normal_(self.att_weight)
        self.att_fc = nn.Linear(2*self.hidden_dim, self.att_len)
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.fc = nn.Linear(2*self.hidden_dim, 2)

        # self.ew = torch.tensor(self.att_weight.data)

    def forward(self, x):
        # print(torch.norm(self.ew - self.att_weight.data)/torch.norm(self.ew))
        x = self.embedding(x)
        h, _ = self.gru(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        hr = h.reshape(h.shape[0]*h.shape[1], h.shape[2])
        u = F.tanh(self.att_fc(hr))
        u = u.reshape(h.shape[0], h.shape[1], -1)
        att_applied = F.softmax(torch.matmul(u, self.att_weight), dim=1)
        h = h.permute(0, 2, 1)
        output = torch.bmm(h, att_applied)
        output = output.squeeze(2)
        self.dropout(output)
        # print(self.att_weight.data[0, 0:10])
        output = self.fc(output)
        return output
