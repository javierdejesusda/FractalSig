
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path
# This script is at projects/fractalsig/scripts/generate_cover.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fractalsig.data_gen import generate_rough_paths

def generate_cover_art():
    """Generates the FractalSig GitHub README image."""
    
    # Configuration 
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  
        'axes.linewidth': 1.5,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.bottom': False,     
        'xtick.labelbottom': False,
        'ytick.left': False,
        'ytick.labelleft': False,
    })

    # Data Generation
    # Generating a long path to show "roughness" effectively
    # standardizing ensures it fits nicely in the plot
    seq_len = 2048 
    H = 0.1
    print(f"Generating Rough Path (H={H})...")
    paths = generate_rough_paths(n_paths=1, seq_len=seq_len, n_channels=1, H=H, standardize=True)
    path = paths[0, :, 0].numpy()

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 5), dpi=300)
    
    ax.plot(path, color='#0077BB', linewidth=1.2, alpha=1.0) # Electric Blue
    
    # Subtle Grid
    ax.grid(True, which='major', color='gray', alpha=0.15, linestyle='-')
    
    # Main Title
    ax.text(0.5, 1.12, r"$\mathrm{FractalSig: Generative\ Rough\ Volatility}$", 
            transform=ax.transAxes, fontsize=28, color='black', 
            verticalalignment='bottom', horizontalalignment='center')
            
    # Subtitle: Calligraphic H
    ax.text(0.5, 1.02, r"$\mathrm{Learned\ Besov-Wavelet\ Decoding}\ \mid\ \mathcal{H} \approx 0.1$", 
            transform=ax.transAxes, fontsize=18, color='#333333',
            verticalalignment='bottom', horizontalalignment='center')

    # Remove margins to make the frame tight around the data but keep the box
    ax.set_xlim(0, seq_len)
    # Add a bit of padding to Y so spikes don't hit the frame
    y_min, y_max = path.min(), path.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Save
    # Save to project_root / results
    output_dir = project_root / 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / 'fractalsig_cover.png'
    
    print(f"Saving cover art to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Done.")

if __name__ == "__main__":
    generate_cover_art()
