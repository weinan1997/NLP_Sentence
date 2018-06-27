import torch
import torch.nn as nn
import torch.nn.functional as F

class SWEM_hier_Sentence(nn.Module):
    
    def __init__(self, args):
        super(SWEM_hier_Sentence, self).__init__()
        self.args = args

        self.kernel_size = 3
        self.stride = 1
        self.fc1 = nn.Linear((args["remain_l"]-self.kernel_size)//self.stride+1, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.kernel_size, stride=self.stride)
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x