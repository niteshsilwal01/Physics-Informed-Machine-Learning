import os
from PIL import Image

# Directory to save frames
output_dir = 'oscillation_frames'
gif_name='Simple Harmonic Oscillator.gif'
os.makedirs(output_dir, exist_ok=True)

def create_gif(output_dir, gif_name= gif_name):
    frames = []
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(output_dir, filename)
            frames.append(Image.open(filepath))

    frames[0].save(os.path.join(output_dir, gif_name), save_all=True, append_images=frames[1:], duration=500, loop=0)