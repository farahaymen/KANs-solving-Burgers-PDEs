import numpy as np

def ema_smoothing(data, alpha=0.999):
    data = np.array(data)
    ema_vals = np.zeros_like(data)
    ema_prev = 0.0

    for i, val in enumerate(data):
        if i == 0:
            ema_prev = val
        else:
            ema_prev = alpha * ema_prev + (1 - alpha) * val
        ema_vals[i] = ema_prev

    return ema_vals
