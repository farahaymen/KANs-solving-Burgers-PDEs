import torch
import numpy as np
import matplotlib.pyplot as plt


def predict(model, x, t, device):
    X = torch.tensor(np.vstack([x, t]).T, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        u = model(X).cpu().numpy().flatten()
    return u


def plot_compare_and_save(model, x, fd_solutions, device, viscosity, outdir):
    """
    Creates and saves 4 plots:
    - t = 0.0
    - t = 0.3
    - t = 0.6
    - t = 1.0
    """
    time_slices = [0.0, 0.3, 0.6, 1.0]

    for t_val in time_slices:
        t_vec = np.full_like(x, t_val, dtype=np.float32)

        u_pred = predict(model, x, t_vec, device)
        u_fd = fd_solutions[t_val]

        plt.figure(figsize=(8, 4))
        plt.plot(x, u_fd, 'b-', label='FD Solution')
        plt.plot(x, u_pred, 'r--', label='Model Prediction')
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"FD vs Model @ t={t_val}, viscosity={viscosity}")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"{outdir}/compare_t_{t_val}.png", dpi=300)
        plt.close()
