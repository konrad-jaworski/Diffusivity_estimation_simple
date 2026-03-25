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
    "/home/kjaworski/Pulpit/Thermal_diffusivity_est/Diffusivity_estimation_simple/data/2026_03_18_CFRP_FBH_3s_30s_top_symetrical.npz",
    allow_pickle=True
)

# -----------------------------
# Patch extraction + conversion
# -----------------------------
T_patch = data['data'][:,190:290,250:350] / 100 - 273.15  

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
T_inf = data['data'][0,190:290,250:350].mean() / 100 - 273.15

# -----------------------------
# Normalize 
# -----------------------------
eps = 1e-8
scale = T_patch.max() - T_inf
T_patch = (T_patch - T_inf) / (scale + eps)

# -----------------------------
# Temporal filtering
# -----------------------------
T_patch = savgol_filter(T_patch, window_length=19, polyorder=3, axis=0)

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

trained_model, l_t, l_d, l_p = train(model, coords, values,Nt,H,W)

# -----------------------------
# Save (FIXED)
# -----------------------------
torch.save(trained_model.state_dict(), 'model_pinn_1.pth')
torch.save(torch.tensor(l_t), 'total_loss_1.pt')
torch.save(torch.tensor(l_d), 'data_loss.pt')
torch.save(torch.tensor(l_p), 'pde_loss.pt')