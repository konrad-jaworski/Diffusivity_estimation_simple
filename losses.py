import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm
import math


def gradients(u, x, order=1):
    """
    Compute first or second derivative of u with respect to x.
    u: [N, 1]
    x: [N, 1] with requires_grad=True
    """
    if order == 1:
        return torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

    elif order == 2:
        du_dx = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        d2u_dx2 = torch.autograd.grad(
            du_dx, x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True
        )[0]

        return d2u_dx2

    else:
        raise ValueError("order must be 1 or 2")
    
def pde_residual_normalized(model, x, y, t, Lx, Ly, Lz, tc):
    """
    PDE residual in normalized coordinates for the reduced surface model:

        theta_t = beta_x * theta_xx + beta_y * theta_yy - beta_z * pi^2 * theta

    where:

        beta_x = alpha_x * tc / Lx^2
        beta_y = alpha_y * tc / Ly^2
        beta_z = alpha_z * tc / Lz^2

    and:

        alpha_i = k_i / (rho * cp)

    The model directly trains physical thermal conductivities:

        kx, ky, kz [W/(m K)]

    Inputs:
        x, y, t : normalized coordinates in [0, 1]
        Lx      : physical x length scale [m]
        Ly      : physical y length scale [m]
        Lz      : physical thickness / z scale [m]
        tc      : physical time scale [s]

    Residual:

        r = theta_t - beta_x*theta_xx - beta_y*theta_yy + beta_z*pi^2*theta
    """

    theta = model(x, y, t)

    theta_t  = gradients(theta, t, order=1)
    theta_xx = gradients(theta, x, order=2)
    theta_yy = gradients(theta, y, order=2)

    # physical diffusivities [m^2/s]
    alpha_x = model.alpha_x()
    alpha_y = model.alpha_y()
    alpha_z = model.alpha_z()

    # convert physical diffusivities to normalized PDE coefficients
    beta_x = alpha_x * tc / (Lx ** 2)
    beta_y = alpha_y * tc / (Ly ** 2)
    beta_z = alpha_z * tc / (Lz ** 2)

    residual = (
        theta_t
        - beta_x * theta_xx
        - beta_y * theta_yy
        + beta_z * (math.pi ** 2) * theta
    )

    return residual

def pde_loss(model, x, y, t,Lx,Ly,Lz,tc):
    r = pde_residual_normalized(model, x, y, t,Lx,Ly,Lz,tc)
    return torch.mean(r ** 2)

#---------------------------------------------- Data loss --------------------------
def data_loss(model, coords, T_true):
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]

    T_pred = model(x, y, t)

    return torch.mean((T_pred - T_true)**2)