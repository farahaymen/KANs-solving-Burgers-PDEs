import torch

def pde_residual(model, X, nu=1.0):
    """
    Burgers' PDE:
        u_t + u u_x - nu u_xx = 0
    """
    X.requires_grad_(True)
    u = model(X)

    grad_u = torch.autograd.grad(
        u, X, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_u_x = torch.autograd.grad(
        u_x, X, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]
    u_xx = grad_u_x[:, 0:1]

    f = u_t + u * u_x - nu * u_xx
    return f

def loss_func(model, X_ic, Y_ic, X_bc, Y_bc, X_coll, nu=1.0):
    u_ic_pred = model(X_ic)
    loss_ic = torch.mean((u_ic_pred - Y_ic)**2)

    u_bc_pred = model(X_bc)
    loss_bc = torch.mean((u_bc_pred - Y_bc)**2)

    f = pde_residual(model, X_coll, nu=nu)
    loss_pde = torch.mean(f**2)

    loss_total = loss_ic + loss_bc + loss_pde
    return loss_total, loss_ic, loss_bc, loss_pde
