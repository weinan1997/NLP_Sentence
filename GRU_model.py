import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_Sentence(nn.Module):
    def __init__(self, args):
        super(GRU_Sentence, self).__init__()
        self.args = args

        self.hidden_dim = args["lstm_dim"]
        self.layer_num = args["layer_num"]
        self.vec_len = args["vec_len"]
        self.dp = args["dropout"]
        self.batch_size = args["batch_size"]
        self.word2vec = args["W"]

        self.embedding = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.gru = nn.GRU(self.vec_len, self.hidden_dim, bidirectional=True)
        self.fc = nn.Linear(2*self.hidden_dim, 2)


    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x.permute(1, 0, 2))
        x = x.permute(1, 2, 0)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        self.dropout(x)
        output = self.fc(x)
        return output
