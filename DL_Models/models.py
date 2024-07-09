import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy

import models
    
class DL_Prediction_Model(nn.Module):
    def __init__(self, input_dim: int, nb_traits, hidden_layer1, hidden_layer2):
       
        super().__init__()
        
        self.net1 = nn.Sequential(
            nn.Linear(in_features = input_dim, out_features = hidden_layer1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer1, out_features = nb_traits)
        )
        
        self.trait2traitnet = nn.Sequential(
            nn.Linear(in_features = nb_traits , out_features = hidden_layer1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer1, out_features = hidden_layer2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer2, out_features = nb_traits)
        )
        
    def forward(self, x):
        x = self.net1(x.float())
        trait2_2 = self.trait2traitnet(x.float().detach())
        return x, trait2_2 
    
class DL_DLGBLUP(nn.Module):
    def __init__(self, nb_traits, hidden_layer1, hidden_layer2):
       
        super().__init__()
        
        self.trait2traitnet = nn.Sequential(
            nn.Linear(in_features = nb_traits , out_features = hidden_layer1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer1, out_features = hidden_layer2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer2, out_features = nb_traits)
        )
        
    def forward(self, x):
        trait2_2 = self.trait2traitnet(x.float())
        return trait2_2 
    
    
    
class MeanTrai2Trait(nn.Module):
    def __init__(self, nb_traits, hidden_layer1, hidden_layer2):

        super().__init__()

        self.meantrait2traitnet = nn.Sequential(
            nn.Linear(in_features = 1 , out_features = hidden_layer1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer1, out_features = hidden_layer2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features = hidden_layer2, out_features = nb_traits)
        )

    def forward(self, x):
        meantrait2 = self.meantrait2traitnet(x.float())
        return meantrait2 