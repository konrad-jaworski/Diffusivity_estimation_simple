import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
from helper_functions import *
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load data
# -----------------------------
sample=np.load("",allow_pickle=True)
X=sample['coord']
Y=sample['temp']

# -----------------------------
# To torch
# -----------------------------
T_patch = torch.from_numpy(T_patch).float().to(device)
Nt, H, W = T_patch.shape
# -----------------------------
# Training data
# -----------------------------

coords = coords.to(device)
values = values.to(device)

# -----------------------------
# Model
# -----------------------------
model = SurfacePINN().to(device)

lr=1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()

loss_total, loss_data, loss_physics = [], [], []
a_x_track, a_y_track, a_z_track = [], [], []

weight=1
lambda_pde = weight

epochs=40000
batch_size=128

for epoch in tqdm(range(epochs)):

    coords_b, values_b = sample_batch(coords, values, batch_size, Nt, H, W, device)

    loss_d = data_loss(model, coords_b, values_b)
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

# -----------------------------
# Save
# -----------------------------
torch.save(trained_model.state_dict(), 'model_pinn_1_laser.pth')
torch.save(torch.tensor(l_t), 'total_loss.pt')
torch.save(torch.tensor(l_d), 'data_loss.pt')
torch.save(torch.tensor(l_p), 'pde_loss.pt')
torch.save(torch.tensor(a_x_track),'a_x_track.pt')
torch.save(torch.tensor(a_y_track),'a_y_track.pt')
torch.save(torch.tensor(a_z_track),'a_z_track.pt')