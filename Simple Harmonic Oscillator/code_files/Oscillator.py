import torch

import PINN_Model
import Analytical_Solution
import Create_GIF
import train

# Parameters
d, w0 = 2, 20
mu, k = 2*d, w0**2

# Initialize the model and optimizer
model = PINN_Model.PINN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training data
x_train = torch.linspace(0, 1, 1000).view(-1, 1).requires_grad_(True)
x_plot = torch.linspace(0, 1, 500).view(-1, 1)
u_analytical = Analytical_Solution.analytical_solution(d, w0, x_plot).float()

print('Training Started !!')
train.train_and_save(model, optimizer, x_train, x_plot, u_analytical, mu, k)
print('Training Ended !!')

# Directory containing saved frames
output_dir = 'oscillation_frames'
gif_name='Simple Harmonic Oscillator.gif'

print(f'Creating {gif_name} GIF file inside {output_dir} .....')
Create_GIF.create_gif(output_dir)
print(f'Successfully created GIF file !!')


