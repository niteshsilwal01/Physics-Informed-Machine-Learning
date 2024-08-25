import torch
import numpy as np

# Compute the analytical solution
def analytical_solution(d, w0, x):
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    u_x  = torch.exp(-d*x)*2*A*cos
    return u_x