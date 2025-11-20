import matplotlib.pyplot as plt
from .smoothing import ema_smoothing
from config import alpha, time_slices, nu

def plot_losses(loss_hist, ic_hist, bc_hist, pde_hist):
    loss_ema = ema_smoothing(loss_hist, alpha=alpha)
    ic_ema = ema_smoothing(ic_hist, alpha=alpha)
    bc_ema = ema_smoothing(bc_hist, alpha=alpha)
    pde_ema = ema_smoothing(pde_hist, alpha=alpha)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_ema, label='Total Loss')
    plt.plot(ic_ema, label='IC Loss')
    plt.plot(bc_ema, label='BC Loss')
    plt.plot(pde_ema, label='PDE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Losses (EMA)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fd_vs_kan(x_fd, fd_solutions, model, kan_predict_fn):
    for t_val in time_slices:
        x_plot = x_fd
        t_plot = (0 * x_plot + t_val).astype('float32')
        u_pred = kan_predict_fn(model, x_plot, t_plot)
        u_actual = fd_solutions[t_val]

        plt.figure(figsize=(8, 4))
        plt.plot(x_plot, u_actual, 'b-', label='FD Actual')
        plt.plot(x_plot, u_pred, 'r-', label='KAN Predicted')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f'Burgers Equation at t = {t_val}, nu={nu} (KAN vs FD)')
        plt.legend()
        plt.grid(True)
        plt.show()
