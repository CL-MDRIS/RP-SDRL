import numpy as np
import torch
import torch.nn as nn

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, ac_space, noise_level=0.1):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.ac_space = ac_space
        self.act_dim = act_dim
        self.noise_level = noise_level

    def forward(self, obs, deterministic=True):
        # Return output from network scaled to action space limits.
        a_theta = self.pi(obs) if deterministic else self.pi(obs) + torch.clamp(
            torch.randn(self.act_dim) * self.noise_level, -0.5, 0.5)
        a_theta = torch.clamp(a_theta, -1, 1)
        a_scale = torch.tensor(self.ac_space.low) + (a_theta + 1.) * 0.5 * (
                    torch.tensor(self.ac_space.high) - torch.tensor(self.ac_space.low))
        return a_scale

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, ac_space, noise_level=0.1,  hidden_sizes=(64, 64, 64, 64),
                 activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, ac_space, noise_level)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            return self.pi(obs, deterministic=deterministic).numpy()
