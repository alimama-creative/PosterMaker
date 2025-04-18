import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAdapterWithLayerNorm(nn.Module):
    def __init__(self, hidden_dim, projection_dim):
        super(LinearAdapterWithLayerNorm, self).__init__()
        self.linear = nn.Linear(hidden_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        # Input first passes through the linear layer
        x = self.linear(x)
        # Then through layer normalization
        x = self.layer_norm(x)
        return x