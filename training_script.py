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
lr = 1e-3
lambda_data = 1.0
lambda_pde = 1.0
lambda_init = 1.0
epochs = 40000
batch_size = 524288
hidden_layers = 3
number_of_neurons = 100

init_log_beta_x = -2.9967
init_log_beta_y = -3.1308
init_log_beta_z = -0.3851

mode = False   # False = freeze diffusivities, True = inverse
subsampling = 2

# Collocation parameters
N_coll = 1000
W_coll = 400
H_coll = 400
subsample_coll = 4

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
Nt, H, W = T_patch.size()

# -----------------------------
# Data generation
# -----------------------------
coords, values = create_training_data(T_patch, subsampling)
coords_init, values_init = create_initial_data(T_patch, subsampling)
coords_coll = create_collocation_points(N_coll, W_coll, H_coll, subsample_coll)

# -----------------------------
# Model
# -----------------------------
model = SurfacePINN(
    hidden_layers=hidden_layers,
    hidden_neurons=number_of_neurons,
    init_log_beta_x=init_log_beta_x,
    init_log_beta_y=init_log_beta_y,
    init_log_beta_z=init_log_beta_z
).to(device)

model.train()

model.log_beta_x.requires_grad = mode
model.log_beta_y.requires_grad = mode
model.log_beta_z.requires_grad = mode

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Tracking
# -----------------------------
loss_total_hist = []
loss_data_hist = []
loss_pde_hist = []
loss_init_hist = []

a_x_track = []
a_y_track = []
a_z_track = []

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

        loss_p = pde_loss(model, x_c, y_c, t_c)

        loss = lambda_data * loss_d + lambda_pde * loss_p + lambda_init * loss_i

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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
        a_x_track.append(model.beta_x().item())
        a_y_track.append(model.beta_y().item())
        a_z_track.append(model.beta_z().item())

    if epoch % 1 == 0:
        print(
            f"[{epoch:5d}] "
            f"L_data={epoch_data:.3e} | "
            f"L_init={epoch_init:.3e} | "
            f"L_pde={epoch_pde:.3e} | "
            f"L_total={epoch_total:.3e} | "
            f"βx={model.beta_x().item():.2e} | "
            f"βy={model.beta_y().item():.2e} | "
            f"βz={model.beta_z().item():.2e}"
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
torch.save(torch.tensor(a_x_track), 'a_x_track.pt')
torch.save(torch.tensor(a_y_track), 'a_y_track.pt')
torch.save(torch.tensor(a_z_track), 'a_z_track.pt')

config = {
    "lr": lr,
    "lambda_data": lambda_data,
    "lambda_pde": lambda_pde,
    "lambda_init": lambda_init,
    "epochs": epochs,
    "batch_size": batch_size,
    "hidden_layers": hidden_layers,
    "number_of_neurons": number_of_neurons,
    "init_log_beta_x": init_log_beta_x,
    "init_log_beta_y": init_log_beta_y,
    "init_log_beta_z": init_log_beta_z,
    "mode": mode,
    "subsampling": subsampling,
    "N_coll": N_coll,
    "W_coll": W_coll,
    "H_coll": H_coll,
    "subsample_coll": subsample_coll,
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)