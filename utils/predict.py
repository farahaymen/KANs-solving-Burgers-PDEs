import numpy as np
import torch
from config import device

def kan_predict(model, x, t):
    """
    x, t: numpy arrays of same shape (N,)
    returns: u_pred as numpy array (N,)
    """
    X = torch.tensor(
        np.hstack([x.reshape(-1, 1), t.reshape(-1, 1)]),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    return u_pred
