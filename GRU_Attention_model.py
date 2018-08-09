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
        self.att_weight = nn.Parameter(torch.rand(2*self.hidden_dim, self.domain_num))
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.fc = nn.Linear(self.seq_len, 2)

    def forward(self, x, z):
        x = self.embedding(x)
        h, _ = self.gru(x.permute(1, 0, 2))
        h = h.permute(1, 0, 2)
        atte_applied = torch.zeros(x.shape[0], self.seq_len)
        if torch.cuda.is_available():
            atte_applied = atte_applied.cuda(self.gpu)
        for row in range(x.shape[0]):
            u = torch.matmul(h[row].squeeze(0), F.softmax(self.att_weight[:, z[row]], dim=0))
            atte_applied[row] = u
        self.dropout(atte_applied)
        output = self.fc(atte_applied)
        return output
