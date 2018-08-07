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

        self.embedding = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.gru = nn.GRU(self.vec_len, self.hidden_dim, bidirectional=True)
        self.att_weight = nn.Parameter(torch.rand(2*self.hidden_dim, 1))
        self.attention = nn.Linear(self.hidden_dim*2, 2*self.hidden_dim)
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.fc = nn.Linear(self.hidden_dim*2, 2)

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.gru(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        u = self.attention(h)
        u = F.tanh(u)
        u = torch.matmul(u, self.att_weight)
        u = F.softmax(u, dim=1)
        u = u.permute(0, 2, 1)
        output = torch.bmm(u, h)
        output = output.squeeze(1)
        self.dropout(output)
        output = self.fc(output)
        return output
