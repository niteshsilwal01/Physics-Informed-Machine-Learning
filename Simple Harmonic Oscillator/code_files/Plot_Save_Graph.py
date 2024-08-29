import matplotlib.pyplot as plt
import os

# Directory to save frames
output_dir = 'oscillation_frames'
os.makedirs(output_dir, exist_ok=True)

def save_frame(epoch, loss, x_plot, u_pinn, u_analytical):
    plt.figure(figsize=(20, 8))
    plt.plot(x_plot.numpy(), u_pinn, label='PINN Prediction', linestyle='-', color='blue')
    plt.plot(x_plot.numpy(), u_analytical, label='Analytical Solution', linestyle='--', color='red')
    plt.xlabel('t', fontdict={'fontsize': 20})
    plt.ylabel('x(t)', fontdict={'fontsize': 20})
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'PINN Model vs. Analytical Solution | Epoch: {epoch} | Loss: {loss.item():.5f}', fontdict={'fontsize': 20})
    plt.legend(fontsize=16)
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, f'frame_{epoch:05d}.png'))
    plt.close()