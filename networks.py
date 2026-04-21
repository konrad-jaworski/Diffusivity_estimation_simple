import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torch.nn as nn

class SurfacePINN(nn.Module):
    """
    Simple PINN network architecture used in our experiment size was selected arbitrary
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 100),  # input: x, y, t
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

        # Xavier initialization
        self.net.apply(self._init_weights)

        # learn diffusivities (log-parametrization since our target variable are close to noise in single point precision)
        self.log_alpha_x = nn.Parameter(torch.tensor([-13.0]))
        self.log_alpha_y = nn.Parameter(torch.tensor([-13.0]))
        self.log_alpha_z = nn.Parameter(torch.tensor([-13.0]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        return self.net(inp)
    
    def retrieve_a(self,a):
        return torch.exp(a)
    