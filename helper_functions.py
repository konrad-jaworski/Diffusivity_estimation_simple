import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter
import torch.nn as nn
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def train(model, coords, values, Nt, H, W, epochs=10000, lr=1e-3, batch_size=32768,weight=1.0):

    

    return model, loss_total, loss_data, loss_physics,a_x_track,a_y_track,a_z_track

