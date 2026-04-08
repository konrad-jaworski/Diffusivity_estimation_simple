import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

L = 3.5e-3  # thickness [m]
pi2_over_L2 = (torch.pi / L)**2 # Our simplification of second derivative in depth

class SurfacePINN(nn.Module):
    """
    Simple PINN network architecture used in our experiment size was selected arbitrary
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128), # We only input x,y,t coordinates as per our simplification
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # learn diffusivities (log-parametrization since our target variable are close to noise in single point precision)
        self.log_alpha_x = nn.Parameter(torch.tensor([-13.0]))
        self.log_alpha_y = nn.Parameter(torch.tensor([-13.0]))
        self.log_alpha_z = nn.Parameter(torch.tensor([-13.0]))

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        return self.net(inp)

    # Methods used to retrieve physical values of diffusion
    def alpha_x(self):
        return torch.exp(self.log_alpha_x)

    def alpha_y(self):
        return torch.exp(self.log_alpha_y)

    def alpha_z(self):
        return torch.exp(self.log_alpha_z)
    

def create_training_data(T_patch):
    # T_patch: [Nt, H, W] used for creating data points for the input to the PINN model

    Nt, H, W = T_patch.shape

    t = torch.linspace(0, 1, Nt)
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)

    tt, yy, xx = torch.meshgrid(t, y, x, indexing='ij')

    coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
    values = T_patch.reshape(-1, 1)

    return coords.to(device), values.to(device)

def _lhs_unit(n, device):
    """
    Latin Hypercube samples in [0,1), one point per bin.
    Returns shape [n].
    """
    u = (torch.arange(n, device=device, dtype=torch.float32) + torch.rand(n, device=device)) / n
    return u[torch.randperm(n, device=device)]


def _to_normalized_coords(t_idx, y_idx, x_idx, Nt, H, W):
    """
    Convert integer grid indices to normalized coordinates in [0,1].
    Output shape: [B, 3] with columns [t, y, x]
    """
    t = t_idx.float() / max(Nt - 1, 1)
    y = y_idx.float() / max(H - 1, 1)
    x = x_idx.float() / max(W - 1, 1)

    coords = torch.stack([t, y, x], dim=1)
    return coords


def sample_data_random_cube(patch, batch_size):
    """
    Random sampling directly from a 3D data cube.

    Parameters
    ----------
    patch : torch.Tensor
        Shape [Nt, H, W]
    batch_size : int

    Returns
    -------
    coords_b : torch.Tensor
        Shape [B, 3], normalized coords [t, y, x]
    values_b : torch.Tensor
        Shape [B, 1]
    """
    device = patch.device
    Nt, H, W = patch.shape

    t_idx = torch.randint(0, Nt, (batch_size,), device=device)
    y_idx = torch.randint(0, H,  (batch_size,), device=device)
    x_idx = torch.randint(0, W,  (batch_size,), device=device)

    coords_b = _to_normalized_coords(t_idx, y_idx, x_idx, Nt, H, W).clone().detach()
    values_b = patch[t_idx, y_idx, x_idx].unsqueeze(1)

    return coords_b, values_b


def sample_data_lhs_cube(patch, batch_size):
    """
    LHS-style sampling directly from a 3D data cube.

    Parameters
    ----------
    patch : torch.Tensor
        Shape [Nt, H, W]
    batch_size : int

    Returns
    -------
    coords_b : torch.Tensor
        Shape [B, 3], normalized coords [t, y, x]
    values_b : torch.Tensor
        Shape [B, 1]
    """
    device = patch.device
    Nt, H, W = patch.shape

    t_idx = torch.clamp((_lhs_unit(batch_size, device) * Nt).long(), 0, Nt - 1)
    y_idx = torch.clamp((_lhs_unit(batch_size, device) * H).long(), 0, H - 1)
    x_idx = torch.clamp((_lhs_unit(batch_size, device) * W).long(), 0, W - 1)

    coords_b = _to_normalized_coords(t_idx, y_idx, x_idx, Nt, H, W).clone().detach()
    values_b = patch[t_idx, y_idx, x_idx].unsqueeze(1)

    return coords_b, values_b


def sample_collocation_random(batch_size, device):
    """
    Random collocation points in normalized continuous domain [0,1]^3.

    Returns
    -------
    coords_f : torch.Tensor
        Shape [B, 3], columns [t, y, x]
    """
    coords_f = torch.rand(batch_size, 3, device=device)
    return coords_f


def sample_collocation_lhs(batch_size, device):
    """
    LHS collocation points in normalized continuous domain [0,1]^3.

    Returns
    -------
    coords_f : torch.Tensor
        Shape [B, 3], columns [t, y, x]
    """
    t = _lhs_unit(batch_size, device)
    y = _lhs_unit(batch_size, device)
    x = _lhs_unit(batch_size, device)

    coords_f = torch.stack([t, y, x], dim=1)
    return coords_f


# def pde_loss(model, coords):

#     coords = coords.clone().detach().requires_grad_(True)

#     x = coords[:, 0:1]
#     y = coords[:, 1:2]
#     t = coords[:, 2:3]

#     T = model(x, y, t) # At this point temperature is between 0 and 1

#     ones = torch.ones_like(T)

#     T_t = torch.autograd.grad(T, t, ones, create_graph=True)[0]

#     T_x = torch.autograd.grad(T, x, ones, create_graph=True)[0]
#     T_xx = torch.autograd.grad(T_x, x, ones, create_graph=True)[0]

#     T_y = torch.autograd.grad(T, y, ones, create_graph=True)[0]
#     T_yy = torch.autograd.grad(T_y, y, ones, create_graph=True)[0]

#     alpha_x = model.alpha_x() # Here we retrieve the diffusivity values encoded in log space, so those are small values
#     alpha_y = model.alpha_y()
#     alpha_z = model.alpha_z()

#     residual = T_t - (
#         alpha_x * T_xx +
#         alpha_y * T_yy -
#         alpha_z * pi2_over_L2 * T
#     )

#     return torch.mean(residual**2)

def pde_loss(model, coords, Nt, H, W):

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

def data_loss(model, coords, T_true):
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]

    T_pred = model(x, y, t)

    return torch.mean((T_pred - T_true)**2)

def sample_batch(coords, values, batch_size, Nt, H, W, device):
    """
    Stratified sampling over (t, x, y)

    coords: [N, 3]
    values: [N, 1]
    """

    # sample indices independently
    t_idx = torch.randint(0, Nt, (batch_size,), device=device)
    x_idx = torch.randint(0, W, (batch_size,), device=device)
    y_idx = torch.randint(0, H, (batch_size,), device=device)

    # map to flattened index
    idx = t_idx * (H * W) + y_idx * W + x_idx

    coords_b = coords[idx].clone().detach()
    values_b = values[idx]

    return coords_b, values_b

def train(model, coords, values, Nt, H, W, epochs=10000, lr=1e-3, batch_size=32768,weight=1.0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    loss_total, loss_data, loss_physics = [], [], []
    a_x_track, a_y_track, a_z_track = [], [], []

    lambda_pde = weight

    for epoch in tqdm(range(epochs)):

        coords_b, values_b = sample_batch(coords, values, batch_size, Nt, H, W, device)

        loss_d = data_loss(model, coords_b, values_b)
        # loss_p = pde_loss(model, coords_b) # Utilize different loss function
        loss_p = pde_loss(model,coords_b,Nt,H,W)
        loss = loss_d + lambda_pde * loss_p

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        loss_total.append(loss.item())
        loss_data.append(loss_d.item())
        loss_physics.append(loss_p.item())
        with torch.no_grad():
            a_x_track.append(model.alpha_x().item())
            a_y_track.append(model.alpha_y().item())
            a_z_track.append(model.alpha_z().item())


        if epoch % 250 == 0:
            print(f"[{epoch:5d}] "
                  f"L_data={loss_d.item():.3e} | "
                  f"L_pde={loss_p.item():.3e} | "
                  f"L_total={loss.item():.3e} | "
                  f"αx={model.alpha_x().item():.2e} | "
                  f"αy={model.alpha_y().item():.2e} | "
                  f"αz={model.alpha_z().item():.2e}")

    model.eval()

    return model, loss_total, loss_data, loss_physics,a_x_track,a_y_track,a_z_track

