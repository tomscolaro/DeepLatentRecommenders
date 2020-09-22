import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepLatentNN(nn.Module):

    def __init__(self, n_users, n_movies, H1 =30, D_out = 10):
        super(DeepLatentNN, self).__init__()
        
        self.layer1 = nn.Embedding(757,50)
        self.layer2 = nn.Linear(50,H1)
        self.layerOut   = nn.Linear(H1, 1)
        



    def forward(self, x):
           x_h =  self.layer1(x)
           x_h2 = self.layer2(x_h)
           preds = self.layerOut(x_h2)

           print(preds)

           return preds





