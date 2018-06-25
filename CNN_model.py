
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
        
        vec_len = args["vec_len"]
        self.filter_num = args["filter_num"]
        kernel_sizes = args["kernel_sizes"]
        kernel_num = args["kernel_num"]
        dropout = args["dropout"]
        
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (k, vec_len)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.filter_num*kernel_num, 2)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        Xs = []
        for i in range(self.filter_num):
            temp = F.relu(self.convs[i](x))
            temp = temp.squeeze(3)
            temp = F.max_pool1d(temp, temp.size(2))
            Xs.append(temp)
        x = torch.cat(Xs, 1)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    


# In[59]:




