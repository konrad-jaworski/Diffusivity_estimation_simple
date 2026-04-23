import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_training_data(T_patch,subsample=2):
    # T_patch: [Nt, H, W] used for creating data points for the input to the PINN model

    Nt, H, W = T_patch.shape

    # Normalization of the coordinates
    t = torch.linspace(0, 1, Nt)
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)

    tt, yy, xx = torch.meshgrid(t, y, x, indexing='ij')

    # stacking them so that each row represents one temperature scalar value
    coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
    values = T_patch.reshape(-1, 1)

    return coords[::subsample,:], values[::subsample,:]

def create_initial_data(T_patch, subsample=2):
    Nt, H, W = T_patch.shape

    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)

    yy, xx = torch.meshgrid(y, x, indexing='ij')
    tt = torch.zeros_like(xx)  # t = 0 for all initial points

    coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
    values = T_patch[0, :, :].reshape(-1, 1)

    return coords[::subsample, :], values[::subsample, :]

def create_collocation_points(Nt=1000,W=400,H=400,subsample=4):
    
    # Normalization of the coordinates
    t = torch.linspace(0, 1, Nt)
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)

    tt, yy, xx = torch.meshgrid(t, y, x, indexing='ij')

    # stacking them so that each row represents one temperature scalar value
    coords = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
    
    return coords[::subsample,:] 

def sample_batch(coords, values, batch_size, idx):
    lower = idx * batch_size
    upper = (idx + 1) * batch_size
    return coords[lower:upper], values[lower:upper]

