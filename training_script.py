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
data = np.load(
    "/home/kjaworski/Pulpit/Thermal_diffusivity_est/Diffusivity_estimation_simple/data/2025_09_12_CFRP_FBH_diffusivity_stationary_p1.npz",
    allow_pickle=True
)

# -----------------------------
# Patch extraction + conversion to C degrees
# -----------------------------
# T_patch = data['data'][:,190:290,250:350] / 100 - 273.15 # Halogen excitation patch
# T_patch = data['data'][:,150:350,230:430] / 100 - 273.15 # Laser excitation patch
T_patch = data['data'][:,225:275,305:355] / 100 - 273.15 # Smaller laser patch

# -----------------------------
# Peak detection for filtering
# -----------------------------
t_signal = T_patch.mean(axis=(1,2))
t_peak_idx = np.argmax(t_signal)

# cut cooling
T_patch = T_patch[t_peak_idx:]

# -----------------------------
# Ambient
# use initial frame before heating
# -----------------------------
# T_inf = data['data'][0,190:290,250:350].mean() / 100 - 273.15
# T_inf = data['data'][0,150:350,230:430].mean() / 100 - 273.15
T_inf = data['data'][:,225:275,305:355].mean() / 100 - 273.15

# -----------------------------
# Normalize 
# -----------------------------
eps = 1e-8
scale = T_patch.max() - T_inf
T_patch = (T_patch - T_inf) / (scale + eps)

# -----------------------------
# Temporal filtering
# -----------------------------
# T_patch = savgol_filter(T_patch, window_length=19, polyorder=3, axis=0) # We do not filter it for the laser heating since we have clean response

# clip safety
T_patch = np.clip(T_patch, 0.0, None)

# -----------------------------
# To torch
# -----------------------------
T_patch = torch.from_numpy(T_patch).float().to(device)
Nt, H, W = T_patch.shape
# -----------------------------
# Training data
# -----------------------------
coords, values = create_training_data(T_patch)
coords = coords.to(device)
values = values.to(device)

# -----------------------------
# Model
# -----------------------------
model = SurfacePINN().to(device)

trained_model, l_t, l_d, l_p , a_x_track, a_y_track, a_z_track= train(model, coords, values,Nt,H,W,epochs=100000)

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