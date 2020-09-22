import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepLatentNN(nn.Module):

    def __init__(self, n_users, n_movies, n_factors, H1, D_out):
        super(DeepLatentNN, self).__init__()
        self.layer = nn
        self.layer = nn



    def forward(self, x):
        prediction_matrix = self.out

        return prediction_matrix

    def predict(self, x):
        # return the score
        output_scores = self.forward(x)
        return output_scores
