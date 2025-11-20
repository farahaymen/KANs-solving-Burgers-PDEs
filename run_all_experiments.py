import torch
import numpy as np
import os

from models.bspline import KAN_BSpline
from models.chebyshev import KAN_Chebyshev
from models.hermite import KAN_Hermite
from models.legendre import KAN_Legendre
from models.pinn import PINN_Burgers

from utils.data import generate_training_data
from utils.losses import pinn_loss
from utils.fd_solver import finite_difference_burgers
from utils.plot import plot_compare_and_save


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

MODELS = {
    "bspline": KAN_BSpline,
    "chebyshev": KAN_Chebyshev,
    "hermite": KAN_Hermite,
    "legendre": KAN_Legendre
}

DEGREES = [3, 5]

VISCOSITIES = [1.0, 0.005]

EPOCHS = 3000
LR = 1e-3


def train_once(model, X_ic, Y_ic, X_bc, Y_bc, X_coll, viscosity):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        loss, lic, lbc, lpde = pinn_loss(
            model,
            X_ic, Y_ic,
            X_bc, Y_bc,
            X_coll,
            viscosity
        )

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  [{epoch}] Loss={loss.item():.4e}")

    return model


def run_all():
    # Load shared training data once
    X_ic, Y_ic, X_bc, Y_bc, X_coll = generate_training_data(
        x_lower=-3, x_upper=3,
        t_lower=0, t_upper=1,
        N_ic=56, N_bc=56, N_coll=10000,
        device=DEVICE
    )

    # Loop through everything
    for model_name, ModelClass in MODELS.items():
        for deg in DEGREES:
            for nu in VISCOSITIES:

                print(f"\n========== Running {model_name.upper()} (degree={deg}, nu={nu}) ==========")

                # Build model
                model = ModelClass(
                    layer_sizes=[2, 64, 64, 64, 64, 1],
                    max_degree=deg if model_name != "bspline" else None,
                    spline_order=deg if model_name == "bspline" else None,
                    domain=[-3, 3]
                ).to(DEVICE)

                # Train
                model = train_once(model, X_ic, Y_ic, X_bc, Y_bc, X_coll, nu)

                # FD reference
                x_fd, sol_fd = finite_difference_burgers(
                    x_lower=-3, x_upper=3,
                    t_max=1.0,
                    nx=401,
                    dt=1e-4,
                    nu=nu
                )

                # Output directory
                outdir = f"results/{model_name}/degree_{deg}/nu_{nu}/"
                os.makedirs(outdir, exist_ok=True)

                # Save plots
                plot_compare_and_save(
                    model=model,
                    x=x_fd,
                    fd_solutions=sol_fd,
                    device=DEVICE,
                    viscosity=nu,
                    outdir=outdir
                )


if __name__ == "__main__":
    run_all()
