import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm


L = 3.5e-3  # thickness [m]
pi2_over_L2 = (torch.pi / L)**2 # Our simplification of second derivative in depth


def pde_loss_primitive(model, coords):

    coords = coords.clone().detach().requires_grad_(True)

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]

    T = model(x, y, t) # At this point temperature is between 0 and 1

    ones = torch.ones_like(T)

    T_t = torch.autograd.grad(T, t, ones, create_graph=True)[0]

    T_x = torch.autograd.grad(T, x, ones, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, ones, create_graph=True)[0]

    T_y = torch.autograd.grad(T, y, ones, create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, ones, create_graph=True)[0]

    alpha_x = model.alpha_x() # Here we retrieve the diffusivity values encoded in log space, so those are small values
    alpha_y = model.alpha_y()
    alpha_z = model.alpha_z()

    residual = T_t - (
        alpha_x * T_xx +
        alpha_y * T_yy -
        alpha_z * pi2_over_L2 * T
    )

    return torch.mean(residual**2)

def pde_loss_physical(model, coords, Nt, H, W):

    coords = coords.clone().detach().requires_grad_(True)

    x = coords[:,0:1]
    y = coords[:,1:2]
    t = coords[:,2:3]

    T = model(x,y,t)

    ones = torch.ones_like(T)

    T_t = torch.autograd.grad(T, t, ones, create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, ones, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, ones, create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, ones, create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, ones, create_graph=True)[0]

    # --- PHYSICAL SCALING ---
    dx = 1e-4 # 100 um
    dy = 1e-4 # 100 um 
    dt = 1/50 # 50 fps

    Lx = dx * W
    Ly = dy * H
    Tmax = dt * Nt

    T_t_phys = T_t / dt
    T_xx_phys = T_xx / (dx**2)
    T_yy_phys = T_yy / (dy**2)


    alpha_x = model.alpha_x()
    alpha_y = model.alpha_y()
    alpha_z = model.alpha_z()

    residual = T_t_phys - (
    alpha_x * T_xx_phys +
    alpha_y * T_yy_phys -
    alpha_z * pi2_over_L2 * T
    )

    return torch.mean(residual**2)

def pde_loss_less(model, coords, Nt, H, W):
    pass



#---------------------------------------------- Data loss --------------------------
def data_loss(model, coords, T_true):
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]

    T_pred = model(x, y, t)

    return torch.mean((T_pred - T_true)**2)