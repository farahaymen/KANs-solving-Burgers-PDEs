import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Domain
x_lower, x_upper = -3.0, 3.0
t_lower, t_upper = 0.0, 1.0

# Data sizes
N_ic = 56
N_bc = 56
N_coll = 10000

# Network architecture
layer_sizes = [2, 64, 64, 64, 64, 1]

# Degrees you want to sweep
degrees = [3, 5]

# Viscosities you want to sweep
viscosities = [1.0, 0.005]

# Training
epochs = 10000
lr = 1e-3
print_every = 1000

# EMA smoothing
alpha = 0.999

# FD solver params (for reference solutions)
nx_fd = 401
dt_fd = 0.0001

# Time slices for comparison
time_slices = [0.0, 0.3, 0.6, 1.0]
