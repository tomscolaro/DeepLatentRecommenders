import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepLatentNN(nn.Module):

    def __init__(self, n_u, H1 =50, D_out = 10):
        super(DeepLatentNN, self).__init__()
        
        self.layer1 = nn.Linear(n_u,10000)
        self.layer2 = nn.Linear(10000,H1)
        self.layerOut   =  nn.Linear(H1, 1)
        



    def forward(self, x):
           x_h =  self.layer1(x)
           x_h2 = F.sigmoid(self.layer2(x_h))
           preds = self.layerOut(x_h2)

           print(preds)

           return preds





