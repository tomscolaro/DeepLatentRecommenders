import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepLatentNN(nn.Module):

    def __init__(self, n_u, H1 =50, D_out = 50):
        super(DeepLatentNN, self).__init__()
        self.layerEncode = nn.Embedding(n_u,50
        self.layer1 = nn.Linear(3,10)
        self.layer2 = nn.Linear(50,H1)
        self.layerOut   =  nn.Linear(H1, 1)
        



    def forward(self, x):  
        
        x_h = self.layerEncode(x)
        #x_h =  torch.cat((self.layerEncode(x[:,0]), self.layer1(x[:,1:])),1)
        x_h2 = F.sigmoid(self.layer2(x_h))
        preds = self.layerOut(x_h2)

        print(preds)

        return preds





