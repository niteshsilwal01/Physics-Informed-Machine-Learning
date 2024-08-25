import torch

# Define the physics-informed loss function
def physics_loss(model, x):
    theta = model(x)

    #dx/dt
    theta_t = torch.autograd.grad(theta, x, torch.ones_like(theta), create_graph=True)[0]

    #dx^2/dt^2
    theta_tt = torch.autograd.grad(theta_t, x, torch.ones_like(theta_t), create_graph=True)[0]

    physics = theta_tt + mu*theta_t + k*theta
    physics = (1e-4)*torch.mean(physics**2)

    return torch.mean(physics**2)