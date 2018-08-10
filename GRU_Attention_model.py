import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_Attention_Sentence(nn.Module):

    def __init__(self, args):
        super(GRU_Attention_Sentence, self).__init__()
        self.args = args
        
        self.hidden_dim = args["attention_dim"]
        self.vec_len = args["vec_len"]
        self.batch_size = args["batch_size"]
        self.word2vec = args["W"]
        self.dp = args["dropout"]
        self.domain_num = args["domain_num"]
        self.seq_len = args["remain_l"]
        self.gpu = args["GPU"]

        self.embedding = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.gru = nn.GRU(self.vec_len, self.hidden_dim, bidirectional=True)
        self.att_weight = nn.Parameter(torch.FloatTensor(self.domain_num, self.hidden_dim*2))
        torch.nn.init.xavier_normal_(self.att_weight)
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.fc = nn.Linear(2*self.hidden_dim, 2)

        # self.ew = torch.tensor(self.att_weight.data)

    def forward(self, x, z):
        # print(torch.norm(self.ew - self.att_weight.data)/torch.norm(self.ew))
        x = self.embedding(x)
        h, _ = self.gru(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        u = torch.matmul(h[0], self.att_weight[z[0]])
        u = u.unsqueeze(0)
        for row in range(1, x.shape[0]):
            temp = torch.matmul(h[row], self.att_weight[z[row]])
            temp = temp.unsqueeze(0)
            u = torch.cat((u, temp))
        att_applied = F.softmax(u, dim=1).view(h.shape[0], h.shape[1], 1)
        u = torch.sum(h*att_applied, dim=1)
        self.dropout(u)
        # print(self.att_weight.data[0, 0:10])
        output = self.fc(u)
        return output
