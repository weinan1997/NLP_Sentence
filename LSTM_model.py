import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Sentence(nn.Module):
    def __init__(self, args):
        super(LSTM_Sentence, self).__init__()
        self.args = args

        self.hidden_dim = args["lstm_dim"]
        self.layer_num = args["layer_num"]
        self.vec_len = args["vec_len"]
        self.dropout = args["dropout"]
        self.batch_size = args["batch_size"]
        self.lstm = nn.LSTM(self.vec_len, self.hidden_dim, self.layer_num)
        self.fc = nn.Linear(self.hidden_dim, 2)


    def forward(self, x):
        x, _ = self.lstm(x.permute(1, 0, 2))
        x = x.permute(1, 2, 0)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        output = self.fc(x)
        return output
