from config import device, epochs, print_every, lr, nu
from data.dataset import generate_training_data
from data.fd_solver import fd_solver
from models.kan_model import KAN
from utils.losses import loss_func
from utils.predict import kan_predict
from utils.plotting import plot_losses, plot_fd_vs_kan

import torch

def main():
    # 1) Data
    X_ic, Y_ic, X_bc, Y_bc, X_coll = generate_training_data()

    # 2) Model
    model = KAN().to(device)

    # 3) Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4) Training loop
    loss_hist = []
    ic_hist = []
    bc_hist = []
    pde_hist = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss_total, loss_ic, loss_bc, loss_pde = loss_func(
            model, X_ic, Y_ic, X_bc, Y_bc, X_coll, nu=nu
        )
        loss_total.backward()
        optimizer.step()

        loss_hist.append(loss_total.item())
        ic_hist.append(loss_ic.item())
        bc_hist.append(loss_bc.item())
        pde_hist.append(loss_pde.item())

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch}: "
                f"Total={loss_total.item():.4e}, "
                f"IC={loss_ic.item():.4e}, "
                f"BC={loss_bc.item():.4e}, "
                f"PDE={loss_pde.item():.4e}"
            )

    # 5) Plot losses
    plot_losses(loss_hist, ic_hist, bc_hist, pde_hist)

    # 6) FD solver + comparison
    x_fd, fd_solutions = fd_solver()
    plot_fd_vs_kan(x_fd, fd_solutions, model, kan_predict)

if __name__ == "__main__":
    main()
