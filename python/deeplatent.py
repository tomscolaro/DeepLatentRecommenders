import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLatentNN(nn.Module):

    def __init__(self, n_users, n_movies, n_factors):
        super(DeepLatentNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name =  torch.cuda.get_device_name(0)
        print('Using {} as device'.format(self.device_name))
        
        self.layerEncodeU = nn.Embedding(n_users ,n_factors,sparse=True)
        self.layerEncodeM = nn.Embedding(n_movies,n_factors,sparse=True)
        self.layerEncodeUB = nn.Embedding(n_users, 1, sparse=True)
        self.layerEncodeMB = nn.Embedding(n_movies, 1, sparse=True)

    def forward(self, x1, x2):

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        preds = self.layerEncodeUB(x1) + self.layerEncodeMB(x2)
        
        preds += (self.layerEncodeU(x1) * self.layerEncodeM(x2)).sum(dim=1, keepdim=True)
        preds = torch.clamp(preds,0,5).cuda()
        return preds.squeeze()
        

#class timeEncodedNN(nn.Module):
    



