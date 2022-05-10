import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class PreNNFunction(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_sizes, activation):
        super().__init__()
        self.y_hat = mlp([in_dim] + list(hidden_sizes) + [out_dim], activation)

    def forward(self, x):
        y_hat = self.y_hat(x)
        return torch.squeeze(y_hat, -1) # Critical to ensure q has right shape.

class PreNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_sizes=(64, 64),
                 activation=nn.ReLU):
        super().__init__()
        self.prenn = PreNNFunction(in_dim, out_dim, hidden_sizes, activation)