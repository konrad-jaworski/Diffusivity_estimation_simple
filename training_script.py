import numpy as np
import torch
from tqdm import tqdm
import json

from helper_functions import (
    create_training_data,
    sample_batch,
    create_initial_data,
    create_collocation_points
)
from losses import pde_loss, data_loss
from networks import SurfacePINN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Training parameters
# -----------------------------
lr_model = 1e-4
lr_param = 1e-6
lambda_data = 1.0
lambda_pde = 1.0
lambda_init = 1.0
epochs = 6000
batch_size = 1024
hidden_layers = 3
number_of_neurons = 100

educated_guess=False # Determine if we want to investigate the exact solution or good guess case

if educated_guess==False:
    # Exact ground truth # [W/(m k)]
    init_kx = 2.23 
    init_ky = 1.95 
    init_kz = 0.93 
elif educated_guess==True:
    init_kx=3
    init_ky=3
    init_kz=3


mode_1 = True  # False = freeze diffusivities, True = inverse for x component
mode_2 = True   # False = freeze diffusivities, True = inverse for y component
mode_3 = True   # False = freeze diffusivities, True = inverse for z component
subsampling = 100
subsampling_init=2

# Collocation parameters
N_coll = 500
W_coll = 200
H_coll = 200
subsample_coll = 100

# -----------------------------
# Load data
# -----------------------------
data = np.load(
    '/home/kjaworski/Pulpit/Thermal_diffusivity_est/Diffusivity_estimation_simple/data/2025_09_12_CFRP_FBH_diffusivity_stationary_p1.npz',
    allow_pickle=True
)

T_patch = data['data'][23:, 150:350, 230:430] / 100 - 273.15
T_amb = data['data'][0:10, 150:350, 230:430].mean(axis=0) / 100 - 273.15
dT = T_patch - T_amb
T_max = dT.max()

T_norm = dT / T_max
T_patch = torch.from_numpy(T_norm).float()

# Required for normalization
Nt, H, W = T_patch.size()

# Normalization parameters
dxy=1e-4 # one pixel is 0.1 mm in spatial coordinates
dt=1/50 # 50 fps is camera recording


Lx=W*dxy # Number of pixels times spatial resolution of the camera
Ly=H*dxy # Number of pixels times spatial resolution of the camera
Lz=3.5e-3 # Thickness of sample is 3.5 [mm]
tc=(Nt-1)*dt # Number of frames times time period

# -----------------------------
# Data generation
# -----------------------------
coords, values = create_training_data(T_patch, subsampling)
coords_init, values_init = create_initial_data(T_patch, subsampling_init)
coords_coll = create_collocation_points(N_coll, W_coll, H_coll, subsample_coll)

print("coords shape:", coords.shape)
print("coords_init shape:", coords_init.shape)
print("coords_coll shape:", coords_coll.shape)

print("coords min:", coords.min(dim=0).values)
print("coords max:", coords.max(dim=0).values)

print("coords_coll min:", coords_coll.min(dim=0).values)
print("coords_coll max:", coords_coll.max(dim=0).values)

print("Physical scales:")
print(f"Lx = {Lx:.6e} m")
print(f"Ly = {Ly:.6e} m")
print(f"Lz = {Lz:.6e} m")
print(f"tc = {tc:.6e} s")

# -----------------------------
# Model
# -----------------------------
model = SurfacePINN(
    hidden_layers=hidden_layers,
    hidden_neurons=number_of_neurons,
    init_kx=init_kx,
    init_ky=init_ky,
    init_kz=init_kz
).to(device)

model.train()

model.kx.requires_grad = mode_1
model.ky.requires_grad = mode_2
model.kz.requires_grad = mode_3

optimizer = torch.optim.Adam(
    [
        {
            "params": model.net.parameters(),
            "lr": lr_model,
        },
        {
            "params": [model.kx, model.ky, model.kz],
            "lr": lr_param,
        },
    ]
)

# -----------------------------
# Tracking
# -----------------------------
loss_total_hist = []
loss_data_hist = []
loss_pde_hist = []
loss_init_hist = []

k_x_track = []
k_y_track = []
k_z_track = []

# separate batch counts
batch_num_data = (coords.size(0) + batch_size - 1) // batch_size
batch_num_init = (coords_init.size(0) + batch_size - 1) // batch_size
batch_num_coll = (coords_coll.size(0) + batch_size - 1) // batch_size

steps_per_epoch = max(batch_num_data, batch_num_init, batch_num_coll)
print("Starting training!")
for epoch in tqdm(range(epochs)):

    loss_total_run = 0.0
    loss_data_run = 0.0
    loss_init_run = 0.0
    loss_pde_run = 0.0

    count_data = 0
    count_init = 0
    count_coll = 0

    for step in tqdm(range(steps_per_epoch)):

        idx_data = step % batch_num_data
        idx_init = step % batch_num_init
        idx_coll = step % batch_num_coll

        # ------------------- sample batches -------------------
        coords_b, values_b = sample_batch(coords, values, batch_size, idx_data)
        coords_init_b, values_init_b = sample_batch(coords_init, values_init, batch_size, idx_init)

        lower_coll = idx_coll * batch_size
        upper_coll = (idx_coll + 1) * batch_size
        coords_coll_b = coords_coll[lower_coll:upper_coll]

        # ------------------- move to device -------------------
        coords_b = coords_b.to(device)
        values_b = values_b.to(device)

        coords_init_b = coords_init_b.to(device)
        values_init_b = values_init_b.to(device)

        coords_coll_b = coords_coll_b.to(device)

        # ------------------- losses -------------------
        loss_d = data_loss(model, coords_b, values_b)
        loss_i = data_loss(model, coords_init_b, values_init_b)

        x_c = coords_coll_b[:, 0:1].clone().detach().requires_grad_(True)
        y_c = coords_coll_b[:, 1:2].clone().detach().requires_grad_(True)
        t_c = coords_coll_b[:, 2:3].clone().detach().requires_grad_(True)

        loss_p = pde_loss(model, x_c, y_c, t_c,Lx,Ly,Lz,tc)

        loss = lambda_data * loss_d + lambda_pde * loss_p + lambda_init * loss_i

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.clamp_conductivities(k_min=0.1, k_max=10.0)
        # ------------------- weighted running averages -------------------
        n_data = values_b.size(0)
        n_init = values_init_b.size(0)
        n_coll = coords_coll_b.size(0)

        loss_data_run += loss_d.item() * n_data
        loss_init_run += loss_i.item() * n_init
        loss_pde_run += loss_p.item() * n_coll

        count_data += n_data
        count_init += n_init
        count_coll += n_coll

    # epoch averages
    epoch_data = loss_data_run / count_data
    epoch_init = loss_init_run / count_init
    epoch_pde = loss_pde_run / count_coll

    epoch_total = (
        lambda_data * epoch_data
        + lambda_pde * epoch_pde
        + lambda_init * epoch_init
    )

    loss_total_hist.append(epoch_total)
    loss_data_hist.append(epoch_data)
    loss_init_hist.append(epoch_init)
    loss_pde_hist.append(epoch_pde)

    with torch.no_grad():
        # if your model uses beta->alpha conversion, adapt this
        k_x_track.append(model.k_x().item())
        k_y_track.append(model.k_y().item())
        k_z_track.append(model.k_z().item())

    if epoch % 1 == 0:
        print(
            f"[{epoch:5d}] "
            f"L_data={epoch_data:.3e} | "
            f"L_init={epoch_init:.3e} | "
            f"L_pde={epoch_pde:.3e} | "
            f"L_total={epoch_total:.3e} | "
            f"kx={model.k_x().item():.2e} | "
            f"ky={model.k_y().item():.2e} | "
            f"kz={model.k_z().item():.2e}"
        )

model.eval()

# -----------------------------
# Save
# -----------------------------
torch.save(model.state_dict(), 'model_pinn_laser.pth')
torch.save(torch.tensor(loss_total_hist), 'total_loss.pt')
torch.save(torch.tensor(loss_data_hist), 'data_loss.pt')
torch.save(torch.tensor(loss_init_hist), 'init_loss.pt')
torch.save(torch.tensor(loss_pde_hist), 'pde_loss.pt')
torch.save(torch.tensor(k_x_track), 'k_x_track.pt')
torch.save(torch.tensor(k_y_track), 'k_y_track.pt')
torch.save(torch.tensor(k_z_track), 'k_z_track.pt')

config = {
    "lr model": lr_model,
    "lr_param": lr_param,
    "lambda_data": lambda_data,
    "lambda_pde": lambda_pde,
    "lambda_init": lambda_init,
    "epochs": epochs,
    "batch_size": batch_size,
    "hidden_layers": hidden_layers,
    "number_of_neurons": number_of_neurons,
    "init_kx": init_kx,
    "init_ky": init_ky,
    "init_kz": init_kz,
    "mode_x": mode_1,
    "mode_y": mode_2,
    "mode_z": mode_3,
    "subsampling": subsampling,
    "subsampling_init":subsampling_init,
    "N_coll": N_coll,
    "W_coll": W_coll,
    "H_coll": H_coll,
    "subsample_coll": subsample_coll,
    "frames":Nt,
    "Height":H,
    "Width":W,
    "Spatial resolution per px [m]": dxy,
    "Temporal resolution [s]": dt,
    "X-scale [m]": Lx,
    "Y-scale [m]": Ly,
    "Z thickness [m]":Lz,
    "Characteristic time":tc,
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)