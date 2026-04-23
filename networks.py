import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SurfacePINN(nn.Module):
    """
    Surface PINN with:
    - normalized inputs: x*, y*, t* in [0, 1]
    - normalized output: theta = (T - T_amb) / DeltaT
    - trainable dimensionless diffusivities beta_x, beta_y, beta_z
      stored in log-space for positivity and numerical stability
    """

    def __init__(self, hidden_layers=3, hidden_neurons=100,
                 init_log_beta_x=-3.5, init_log_beta_y=-3.5, init_log_beta_z=0.0):
        super().__init__()

        layers = []

        # first hidden layer
        layers.append(nn.Linear(3, hidden_neurons))
        layers.append(nn.Tanh())

        # remaining hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())

        # output layer
        layers.append(nn.Linear(hidden_neurons, 1))

        self.net = nn.Sequential(*layers)
        self.net.apply(self._init_weights)

        # train dimensionless diffusivities in log-space
        self.log_beta_x = nn.Parameter(torch.tensor([init_log_beta_x], dtype=torch.float32))
        self.log_beta_y = nn.Parameter(torch.tensor([init_log_beta_y], dtype=torch.float32))
        self.log_beta_z = nn.Parameter(torch.tensor([init_log_beta_z], dtype=torch.float32))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        return self.net(inp)

    # dimensionless diffusivities
    def beta_x(self):
        return torch.exp(self.log_beta_x)

    def beta_y(self):
        return torch.exp(self.log_beta_y)

    def beta_z(self):
        return torch.exp(self.log_beta_z)

    # recover physical diffusivities [m^2/s]
    def alpha_x(self, Lx, tc):
        return self.beta_x() * (Lx ** 2) / tc

    def alpha_y(self, Ly, tc):
        return self.beta_y() * (Ly ** 2) / tc

    def alpha_z(self, Lz, tc):
        return self.beta_z() * (Lz ** 2) / tc

    # convenient scalar values for logging
    def alpha_x_value(self, Lx, tc):
        return self.alpha_x(Lx, tc).detach().cpu().double().item()

    def alpha_y_value(self, Ly, tc):
        return self.alpha_y(Ly, tc).detach().cpu().double().item()

    def alpha_z_value(self, Lz, tc):
        return self.alpha_z(Lz, tc).detach().cpu().double().item()

    def beta_x_value(self):
        return self.beta_x().detach().cpu().double().item()

    def beta_y_value(self):
        return self.beta_y().detach().cpu().double().item()

    def beta_z_value(self):
        return self.beta_z().detach().cpu().double().item()