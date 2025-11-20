import torch
import torch.nn as nn

class PINN_Burgers(nn.Module):
    """
    Physics-Informed Neural Network wrapper for Burgers' equation.
    Works with ANY neural model that maps (x, t) -> u.
    """

    def __init__(self, model, viscosity=1.0):
        super().__init__()
        self.model = model          # KAN model (B-spline / Chebyshev / Hermite / Legendre)
        self.nu = viscosity         # viscosity (1.0 or 0.005)

    def forward(self, X):
        return self.model(X)

    def residual(self, X):
        """
        Compute PDE residual for Burgers' equation:
            u_t + u u_x - nu * u_xx = 0
        """
        nu = self.nu
        X.requires_grad_(True)

        u = self.model(X)

        # First derivatives
        grad_u = torch.autograd.grad(
            u, X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        # Second derivative
        grad_u_x = torch.autograd.grad(
            u_x, X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx = grad_u_x[:, 0:1]

        # PDE residual
        f = u_t + u * u_x - nu * u_xx
        return f

    def loss(self, X_ic, Y_ic, X_bc, Y_bc, X_coll):
        """
        Compute total loss:
            MSE(IC) + MSE(BC) + MSE(PDE residual)
        """

        # Initial condition u(x, 0)
        ic_pred = self.model(X_ic)
        loss_ic = torch.mean((ic_pred - Y_ic)**2)

        # Boundary condition u(x_left, t), u(x_right, t)
        bc_pred = self.model(X_bc)
        loss_bc = torch.mean((bc_pred - Y_bc)**2)

        # PDE collocation points
        f = self.residual(X_coll)
        loss_pde = torch.mean(f**2)

        loss_total = loss_ic + loss_bc + loss_pde
        return loss_total, loss_ic, loss_bc, loss_pde
