import numpy as np
from config import *

def fd_solver():
    x = np.linspace(x_lower, x_upper, nx_fd)
    dx = x[1] - x[0]

    u = np.exp(-x**2)
    u[0] = 0.0
    u[-1] = 0.0

    sol = {}
    curr_time = 0.0
    save_idx = 0
    nt = int(t_upper / dt_fd)

    for step in range(nt + 1):
        if save_idx < len(time_slices) and abs(curr_time - time_slices[save_idx]) < dt_fd/2:
            sol[time_slices[save_idx]] = u.copy()
            save_idx += 1

        un = u.copy()
        u[1:-1] = (
            un[1:-1]
            - dt_fd * un[1:-1] * (un[2:] - un[:-2])/(2*dx)
            + dt_fd * nu * (un[2:] - 2*un[1:-1] + un[:-2])/(dx**2)
        )

        u[0] = 0.0
        u[-1] = 0.0
        curr_time += dt_fd

    return x, sol
