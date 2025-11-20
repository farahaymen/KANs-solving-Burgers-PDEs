import numpy as np
import torch
from config import *

def generate_training_data():
    # Initial condition
    x_ic = np.random.uniform(x_lower, x_upper, (N_ic, 1))
    t_ic = np.zeros((N_ic, 1))
    u_ic = np.exp(-x_ic**2)

    # Boundary
    t_bc_r = np.random.uniform(t_lower, t_upper, (N_bc, 1))
    x_bc_l = np.ones((N_bc // 2, 1)) * x_lower
    x_bc_r = np.ones((N_bc - N_bc // 2, 1)) * x_upper

    t_bc_l = t_bc_r[:N_bc//2]
    t_bc_r2 = t_bc_r[N_bc//2:]

    u_bc_l = np.zeros_like(x_bc_l)
    u_bc_r = np.zeros_like(x_bc_r)

    x_bc = np.vstack([x_bc_l, x_bc_r])
    t_bc = np.vstack([t_bc_l, t_bc_r2])
    u_bc = np.vstack([u_bc_l, u_bc_r])

    # Collocation
    x_col = np.random.uniform(x_lower, x_upper, (N_coll, 1))
    t_col = np.random.uniform(t_lower, t_upper, (N_coll, 1))

    X_ic = torch.tensor(np.hstack([x_ic, t_ic]), dtype=torch.float32, device=device)
    Y_ic = torch.tensor(u_ic, dtype=torch.float32, device=device)

    X_bc = torch.tensor(np.hstack([x_bc, t_bc]), dtype=torch.float32, device=device)
    Y_bc = torch.tensor(u_bc, dtype=torch.float32, device=device)

    X_coll = torch.tensor(np.hstack([x_col, t_col]), dtype=torch.float32, device=device)

    return X_ic, Y_ic, X_bc, Y_bc, X_coll
