import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn

class PointwiseCBM(nn.Module):
    def __init__(self, n_features, n_concepts, hidden_dim, output_dim):
        super().__init__()
        self.n_features = n_features
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.concept_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_concepts*output_dim),
            #nn.Sigmoid()  # concepts typically in [0,1]
        )
        
        self.output_net = nn.Linear(n_concepts*output_dim, output_dim)

    def forward(self, x):
        B, V, T, Y, X = x.shape
        x = x.permute(0, 3, 4, 1, 2)      # (B, Y, X, T, V)
        x = x.reshape(B * Y * X, T * V)  # flatten spatial points
        
        concepts = self.concept_net(x)    # (B*Y*X, n_concepts)
        concepts = concepts.view(B * Y * X, self.output_dim * self.n_concepts)
        
        output = self.output_net(concepts) # (B*Y*X, output_dim)
        
        # reshape back to grid
        concepts = concepts.view(B, Y, X, self.output_dim, self.n_concepts)
        output = output.view(B, Y, X, self.output_dim, -1)

        concepts = concepts.permute(0, 3, 1, 2, 4)  # (B, lead, Y, X, n_concepts)
        output = output.permute(0, 3, 1, 2, 4)      # (B, lead, Y, X, output_dim)
        
        return output, concepts