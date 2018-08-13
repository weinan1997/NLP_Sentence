import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ## Functions to accomplish attention

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()



def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()



def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)


# ## Word attention model with bias

class AttentionWordRNN(nn.Module):
    
    
    def __init__(self, args):        
        
        super(AttentionWordRNN, self).__init__()
        
        self.word_gru_hidden = args["attention_dim"]
        self.word2vec = args["W"]
        self.vec_len = args["vec_len"]
        self.batch_size = args["batch_size"]

        
        self.lookup = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.lookup.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.word_gru = nn.GRU(self.vec_len, self.word_gru_hidden, bidirectional= True)
        self.weight_W_word = nn.Parameter(torch.Tensor(2* self.word_gru_hidden,2*self.word_gru_hidden))
        self.bias_word = nn.Parameter(torch.Tensor(2* self.word_gru_hidden,1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(2*self.word_gru_hidden, 1))
        self.fc = nn.Linear(2* self.word_gru_hidden, 2)
            
        self.softmax_word = nn.Softmax(dim=0)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)

        
        
    def forward(self, embed):
        # embeddings
        embedded = self.lookup(embed)
        embedded = embedded.permute(1, 0, 2)
        # word level gru
        output_word, _ = self.word_gru(embedded)
#         print output_word.size()
        word_squish = batch_matmul_bias(output_word, self.weight_W_word,self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))
        output = self.fc(word_attn_vectors)
            
        return output 

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))