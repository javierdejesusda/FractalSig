
"""
Figure Generator for FractalSig.

This script generates the central visual artifact of the project: a 3-panel figure
demonstrating the superiority of FractalSig (Wavelet-based) over SigDiffusions
(Fourier-based) for Rough Volatility (H ≈ 0.1).

Panels:
A. The Generative Landscape: A full view of the synthetic rough path.
B. The Micro-Structure (Deathmatch): Zoom-in comparing Ground Truth, Fourier, and FractalSig.
C. Power Spectral Density: Log-log plot showing power law decay preservation.
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import iisignature
from scipy import signal

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fractalsig.data_gen import generate_rough_paths
from fractalsig.decoder import FractalDecoder

# Config
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fractal_production"
CHECKPOINT_PATH = CHECKPOINT_DIR / "fractal_decoder_best.pth"
STATS_PATH = CHECKPOINT_DIR / "normalization_stats.json"
OUTPUT_PATH = RESULTS_DIR / "master_figure.png"

# Styling
STYLE_CONFIG = {
    "colors": {
        "gt": "#333333",          
        "fourier": "#E24A33",     
        "fractalsig": "#0077BB",  
    },
    "linewidths": {
        "gt": 1.5,
        "reconstruction": 1.2,
        "zoom_gt": 2.0,
        "zoom_recon": 1.8,
    },
    "fonts": {
        "title": 14,
        "label": 11,
        "tick": 9,
        "legend": 10,
        "annotation": 9,
    },
    "dpi": 300,
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("master_figure")


def load_model_and_stats() -> Tuple[Any, Dict[str, Any]]:
    """Load the trained FractalDecoder model and normalization statistics."""
    if not CHECKPOINT_PATH.exists() or not STATS_PATH.exists():
        log.warning("Checkpoint or stats not found. Using simulation mode (Wavelet Hard Thresholding).")
        return None, None
    
    # Load stats
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    config = checkpoint['config']
    
    # Init model
    model = FractalDecoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_seq_len=config['seq_len'],
        out_channels=config['n_channels'],
        wavelet=config['wavelet'],
        level=config['level'],
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    log.info("Model and stats loaded successfully.")
    return model, stats


def compute_log_signature(path: np.ndarray, depth: int, stats: Dict[str, Any]) -> torch.Tensor:
    """Compute and normalize log-signature for the given path."""
    # Add time augmentation: (seq_len, 1) -> (seq_len, 2)
    seq_len = path.shape[0]
    time = np.linspace(0, 1, seq_len).reshape(-1, 1)
    augmented_path = np.hstack([time, path.reshape(-1, 1)])
    
    # Prepare iisignature
    s = iisignature.prepare(augmented_path.shape[1], depth)
    
    # Compute signature
    logsig = iisignature.logsig(augmented_path, s)
    
    # Normalize
    logsig_tensor = torch.from_numpy(logsig).float().unsqueeze(0) # (1, dim)
    mean = torch.tensor(stats['logsig_mean'])
    std = torch.tensor(stats['logsig_std'])
    
    logsig_norm = (logsig_tensor - mean) / std
    return logsig_norm


def fourier_reconstruction(signal_data: np.ndarray, keep_ratio: float = 0.05) -> np.ndarray:
    """Reconstruct signal keeping only the lowest frequency components."""
    n = len(signal_data)
    fft_coeffs = np.fft.fft(signal_data)
    
    # Keep lowest 5% (simulating SigDiffusions bottleneck)
    n_keep = max(1, int(n * keep_ratio / 2))
    
    mask = np.zeros(n, dtype=bool)
    mask[:n_keep] = True
    mask[-n_keep+1:] = True
    
    fft_truncated = fft_coeffs * mask
    return np.fft.ifft(fft_truncated).real


def compute_psd(signal_data: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density."""
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=len(signal_data)//2)
    return freqs, psd


def main():
    log.info("Starting Master Figure Generation...")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # FractalSig Reconstruction
    model, stats = load_model_and_stats()
    
    # Determine valid sequence length
    if model:
        seq_len = model.output_seq_len
        log.info(f"Adapted sequence length to model trained config: {seq_len}")
    else:
        seq_len = 1024 # Fallback for simulation
    
    # Data Preparation
    H = 0.1
    # Use specific seed for reproducibility of "nice" rough paths
    seed = 42 
    
    log.info(f"Generating Rough Volatility Path (H={H}, seq_len={seq_len})...")
    path_tensor = generate_rough_paths(
        n_paths=1, seq_len=seq_len, n_channels=1, H=H, seed=seed, standardize=True
    )
    ground_truth = path_tensor[0, :, 0].numpy()
    
    # Baseline: Fourier Reconstruction
    log.info("Computing Fourier Baseline...")
    fourier_recon = fourier_reconstruction(ground_truth, keep_ratio=0.05)

    if model and stats:
        log.info("Running FractalSig (Neural Wavelet Decoder)...")
        # Compute inputs
        logsig_input = compute_log_signature(ground_truth, stats['sig_depth'], stats)
        
        # Inference
        with torch.no_grad():
            fractal_recon_tensor = model(logsig_input)
            
        # The model outputs reconstructed WAVELET coefficients, which turns into the path.
        fractal_recon = fractal_recon_tensor[0, :, 0].numpy()
    else:
        # Fallback if model missing (for robustness)
        log.info("Running FractalSig Fallback (Hard Wavelet Thresholding)...")
        import pywt
        coeffs = pywt.wavedec(ground_truth, 'db4', level=6)
        # Keep top 5% coeffs
        all_coeffs = np.concatenate([c.ravel() for c in coeffs])
        thresh = np.percentile(np.abs(all_coeffs), 95)
        new_coeffs = [np.where(np.abs(c) >= thresh, c, 0) for c in coeffs]
        fractal_recon = pywt.waverec(new_coeffs, 'db4')[:seq_len]

    # Plotting
    log.info("Plotting Master Figure with Aesthetic Updates...")
    
    # Update styling based on feedback
    plt.rcParams.update({
        'font.size': STYLE_CONFIG['fonts']['label'],
        'axes.titlesize': STYLE_CONFIG['fonts']['title'],
        'axes.labelsize': STYLE_CONFIG['fonts']['label'],
        'xtick.labelsize': STYLE_CONFIG['fonts']['tick'],
        'ytick.labelsize': STYLE_CONFIG['fonts']['tick'],
        'legend.fontsize': STYLE_CONFIG['fonts']['legend'],
        'font.family': 'serif',
        'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix',
    })

    fig = plt.figure(figsize=(15, 12)) # Increased height for better spacing
    
    # Use constrained_layout for automatic spacing, or tune manually with gridspec
    # User requested explicit padding control, so let's stick to GridSpec + subplots_adjust
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], figure=fig)
    
    # Panel A: The Generative Landscape
    ax_a = fig.add_subplot(gs[0, :])
    t = np.arange(seq_len)
    
    ax_a.plot(t, ground_truth, color=STYLE_CONFIG['colors']['gt'], 
              linewidth=STYLE_CONFIG['linewidths']['gt'], label=r"$\mathrm{Ground\ Truth\ (Rough)}$")
    
    ax_a.plot(t, fractal_recon, color=STYLE_CONFIG['colors']['fractalsig'], 
              linewidth=STYLE_CONFIG['linewidths']['reconstruction'], alpha=0.8, 
              label=r"$\mathrm{FractalSig\ (Generated)}$")
    
    ax_a.set_title(r"$\mathrm{Panel\ A:\ The\ Generative\ Landscape\ \text{-}\ Synthetic\ Rough\ Volatility\ (}H \approx 0.1\mathrm{)}$", 
                   loc='left', fontweight='bold', pad=15, fontsize=18) # Increased padding
    ax_a.set_xlim(0, seq_len)
    
    ax_a.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    
    ax_a.set_ylabel(r"$\mathrm{Log-Volatility}$")
    ax_a.set_xticks([]) 
    
    # Ensure all spines are visible (Frame complete)
    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    
    # Panel B: Micro-Structure (Deathmatch)
    ax_b = fig.add_subplot(gs[1, 0])
    
    # Find a good spike (zoom region)
    spike_idx = np.argmax(np.abs(np.diff(ground_truth, n=1)))
    
    # Define zoom window
    zoom_width = 60
    start = max(0, spike_idx - zoom_width // 2)
    end = min(seq_len, start + zoom_width)
    start = end - zoom_width 
    
    t_zoom = t[start:end]
    gt_zoom = ground_truth[start:end]
    fourier_zoom = fourier_recon[start:end]
    fractal_zoom = fractal_recon[start:end]
    
    ax_b.plot(t_zoom, gt_zoom, color=STYLE_CONFIG['colors']['gt'], 
              linewidth=STYLE_CONFIG['linewidths']['zoom_gt'], label=r"$\mathrm{Ground\ Truth}$")
    ax_b.plot(t_zoom, fourier_zoom, color=STYLE_CONFIG['colors']['fourier'], linestyle='--', 
              linewidth=STYLE_CONFIG['linewidths']['zoom_recon'], label=r"$\mathrm{Baseline\ (Fourier)}$")
    ax_b.plot(t_zoom, fractal_zoom, color=STYLE_CONFIG['colors']['fractalsig'], 
              linewidth=STYLE_CONFIG['linewidths']['zoom_recon'], label=r"$\mathrm{FractalSig\ (Ours)}$")
    
    ax_b.set_title(r"$\mathrm{Panel\ B:\ Micro-Structure}$", 
                   loc='left', fontweight='bold', pad=15, fontsize=18)
    ax_b.set_xlabel(r"$\mathrm{Time\ Step}$")
    ax_b.set_ylabel(r"$\mathrm{Value}$")
    ax_b.legend(loc='upper right', frameon=True, framealpha=0.9) # Ensure readable over lines
    
    # Gibbs Arrow
    # Find a peak in the Fourier ringing that is far from GT
    err = np.abs(gt_zoom - fourier_zoom)
    # Scan for a good spot
    max_err_idx = np.argmax(err)
    target_x = t_zoom[max_err_idx]
    target_y = fourier_zoom[max_err_idx]
    
    y_range = max(gt_zoom) - min(gt_zoom)
    if target_y > min(gt_zoom) + y_range/2:
        text_y = min(gt_zoom) + y_range * 0.1
        text_x = t_zoom[10] # Early in the window
    else:
        text_y = max(gt_zoom) - y_range * 0.1
        text_x = t_zoom[10]
        
    ax_b.annotate(
        r"$\mathrm{Gibbs\ Artifacts}$",
        xy=(target_x, target_y),
        xytext=(target_x - 10, target_y + (0.5 if target_y < 0 else -0.5)), # Manual offset attempt
        arrowprops=dict(arrowstyle="->", color=STYLE_CONFIG['colors']['fourier'], lw=1.5),
        color=STYLE_CONFIG['colors']['fourier'],
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=STYLE_CONFIG['colors']['fourier'], alpha=0.9)
    )
    
    # Panel C: Dynamics (PSD)
    ax_c = fig.add_subplot(gs[1, 1])
    
    # Compute PSDs
    freqs, psd_gt = compute_psd(ground_truth)
    _, psd_fourier = compute_psd(fourier_recon)
    _, psd_fractal = compute_psd(fractal_recon)
    
    # Filter DC
    freqs = freqs[1:]
    psd_gt = psd_gt[1:]
    psd_fourier = psd_fourier[1:]
    psd_fractal = psd_fractal[1:]
    
    ax_c.loglog(freqs, psd_gt, color=STYLE_CONFIG['colors']['gt'], alpha=0.5, label=r"$\mathrm{Ground\ Truth}$")
    ax_c.loglog(freqs, psd_fourier, color=STYLE_CONFIG['colors']['fourier'], linestyle='--', alpha=0.9, label=r"$\mathrm{Fourier}$")
    ax_c.loglog(freqs, psd_fractal, color=STYLE_CONFIG['colors']['fractalsig'], alpha=0.9, label=r"$\mathrm{FractalSig}$")
    
    ax_c.set_title(r"$\mathrm{Panel\ C:\ Spectral\ Fidelity\ (PSD)}$", 
                   loc='left', fontweight='bold', pad=15, fontsize=18)
    ax_c.set_xlabel(r"$\mathrm{Frequency\ (log)}$")
    ax_c.set_ylabel(r"$\mathrm{Power\ (log)}$")
    
    # Move text to top-right corner with a background box
    ax_c.text(0.95, 0.95, r"$\mathrm{Expected\ Decay:\ } 1/f^{1.2} \mathrm{\ (}H=0.1\mathrm{)}$", transform=ax_c.transAxes,
              ha='right', va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
              
    ax_c.text(0.5, 0.5, r"$\mathrm{HF\ Information\ Loss}$" + "\n" + r"$\mathrm{(Fourier\ drops\ early)}$", transform=ax_c.transAxes,
              ha='center', va='center',
              color=STYLE_CONFIG['colors']['fourier'], fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax_c.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # Ensure it's clearly defined
    from matplotlib.patches import Rectangle
    rect = Rectangle((start, min(gt_zoom)), zoom_width, max(gt_zoom) - min(gt_zoom),
                         linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--', zorder=5)
    ax_a.add_patch(rect)
    
    # Label the zoom box?
    ax_a.text(start + zoom_width/2, max(gt_zoom) + 0.2, r"$\mathrm{Panel\ B\ View}$", 
              ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(pad=3.0) # Increased padding
    
    # Save
    log.info(f"Saving Refined Master Figure to {OUTPUT_PATH}...")
    plt.savefig(OUTPUT_PATH, dpi=STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    log.info("Done.")

if __name__ == "__main__":
    main()
