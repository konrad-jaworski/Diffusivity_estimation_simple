import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torch.nn as nn


class SurfacePINN(nn.Module):
    """
    Surface PINN with:
    - normalized inputs: x*, y*, t* in [0, 1]
    - normalized output: theta = (T - T_amb) / DeltaT
    - trainable physical thermal conductivities:
        kx, ky, kz [W/(m K)]
    - fixed density rho [kg/m^3]
    - fixed specific heat cp [J/(kg K)]

    Thermal diffusivities are recovered internally as:

        alpha_i = k_i / (rho * cp)

    with alpha_i in [m^2/s].
    """

    def __init__(
        self,
        hidden_layers=3,
        hidden_neurons=100,
        init_kx=2.23,
        init_ky=1.95,
        init_kz=0.93,
        rho=1600.0,
        cp=700.0,
    ):
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

        # train physical conductivities directly [W/(m K)]
        self.kx = nn.Parameter(torch.tensor([init_kx], dtype=torch.float32))
        self.ky = nn.Parameter(torch.tensor([init_ky], dtype=torch.float32))
        self.kz = nn.Parameter(torch.tensor([init_kz], dtype=torch.float32))

        # fixed material properties
        self.register_buffer("rho", torch.tensor(float(rho), dtype=torch.float32))
        self.register_buffer("cp", torch.tensor(float(cp), dtype=torch.float32))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        return self.net(inp)

    # --------------------------------------------------
    # Thermal conductivities [W/(m K)]
    # --------------------------------------------------

    def k_x(self):
        return self.kx

    def k_y(self):
        return self.ky

    def k_z(self):
        return self.kz

    # --------------------------------------------------
    # Thermal diffusivities [m^2/s]
    # alpha = k / (rho * cp)
    # --------------------------------------------------

    def alpha_x(self):
        return self.kx / (self.rho * self.cp)

    def alpha_y(self):
        return self.ky / (self.rho * self.cp)

    def alpha_z(self):
        return self.kz / (self.rho * self.cp)

    # --------------------------------------------------
    # Convenient scalar values for logging
    # --------------------------------------------------

    def k_x_value(self):
        return self.kx.detach().cpu().double().item()

    def k_y_value(self):
        return self.ky.detach().cpu().double().item()

    def k_z_value(self):
        return self.kz.detach().cpu().double().item()

    def alpha_x_value(self):
        return self.alpha_x().detach().cpu().double().item()

    def alpha_y_value(self):
        return self.alpha_y().detach().cpu().double().item()

    def alpha_z_value(self):
        return self.alpha_z().detach().cpu().double().item()

    def rho_value(self):
        return self.rho.detach().cpu().double().item()

    def cp_value(self):
        return self.cp.detach().cpu().double().item()

    # --------------------------------------------------
    # Optional safety clamp after optimizer step
    # --------------------------------------------------

    def clamp_conductivities(self, k_min=1e-6, k_max=20.0):
        """
        Optional: keep conductivities physically positive.

        k_min, k_max units: [W/(m K)]
        """
        with torch.no_grad():
            self.kx.clamp_(k_min, k_max)
            self.ky.clamp_(k_min, k_max)
            self.kz.clamp_(k_min, k_max)