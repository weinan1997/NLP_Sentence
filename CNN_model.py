
# coding: utf-8

# In[60]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[61]:


class CNN_Sentence(nn.Module):
    
    def __init__(self, args):
        super(CNN_Sentence, self).__init__()
        self.args = args
        
        self.vec_len = args["vec_len"]
        self.kernel_sizes = args["kernel_sizes"]
        self.filter_num = len(self.kernel_sizes)
        self.kernel_num = args["kernel_num"]
        self.dp = args["dropout"]
        self.word2vec = args["W"]

        self.embedding = nn.Embedding(self.word2vec.shape[0], self.vec_len)
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.vec_len)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dp, inplace=True)
        self.fc = nn.Linear(self.filter_num*self.kernel_num, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        Xs = []
        for i in range(self.filter_num):
            temp = F.relu(self.convs[i](x))
            temp = temp.squeeze(3)
            temp = F.max_pool1d(temp, temp.size(2))
            Xs.append(temp)
        x = torch.cat(Xs, 1)
        x = x.squeeze(2)
        self.dropout(x)
        x = self.fc(x)
        return x
    


# In[59]:




