import torch
import torch.nn as nn

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()

        self.fci = nn.Linear(n_input, n_hidden)
        
        self.fch = nn.Sequential(*[
                                nn.Sequential(*[
                                                nn.Linear(n_hidden, n_hidden),
                                                nn.Tanh()]) 
                                                for _ in range(n_layers-1)])
        
        self.fco = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.tanh(self.fci(x))
        x = self.fch(x)
        x = self.fco(x)
        return x