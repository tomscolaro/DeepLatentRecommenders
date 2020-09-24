import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepLatentNN(nn.Module):

    def __init__(self, n_users, n_movies, n_factors):
        super(DeepLatentNN, self).__init__()
        self.layerEncodeU = nn.Embedding(n_users ,n_factors,sparse=True)
        self.layerEncodeM = nn.Embedding(n_movies,n_factors,sparse=True)
        self.layerEncodeUB = torch.nn.Embedding(n_users, 1, sparse=True)
        self.layerEncodeMB = torch.nn.Embedding(n_movies, 1, sparse=True)

    
    def forward(self, x1, x2):  
        preds = self.layerEncodeUB(x1) + self.layerEncodeMB(x2)
        preds += (self.layerEncodeU(x1) * self.layerEncodeM(x2)).sum(dim=1, keepdim=True)
        return preds.squeeze()
        





