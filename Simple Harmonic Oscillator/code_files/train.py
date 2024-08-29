import torch
from IPython.display import clear_output

import Physics_Informed_Loss
import Plot_Save_Graph

n_epochs = 10000

# Training the model
def train_and_save(model, optimizer, x_train, x_plot, u_analytical, mu, k, epochs=n_epochs, plot_interval=200):
    for epoch in range(epochs):

        optimizer.zero_grad()
        y_p = model(x_plot)

        data_loss_val = torch.mean((u_analytical-y_p)**2)
        physics_loss_val = Physics_Informed_Loss.physics_loss(model, x_train, mu, k)

        loss = data_loss_val + physics_loss_val

        loss.backward()
        optimizer.step()

        if epoch % plot_interval == 0:
            clear_output(wait=True)
            u_pinn = model(x_plot).detach().numpy()
            Plot_Save_Graph.save_frame(epoch + plot_interval, loss, x_plot, u_pinn, u_analytical)
            print(f'Epoch: {epoch} | Loss: {loss.item():.5f}')