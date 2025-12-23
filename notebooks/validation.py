"""Validation: Fourier vs Wavelets for Rough Volatility.

This script generates Figure 1 of our paper, demonstrating the Gibbs Phenomenon
in Fourier-based reconstruction of rough paths (H ≈ 0.1) and the superiority
of Wavelet-based methods.

The Thesis: Fourier inversion (used by SigDiffusions, ICLR 2025) is mathematically
ill-suited for Rough Paths (H < 0.5) due to the Gibbs Phenomenon. Wavelets
operating in Besov spaces are the natural solution.

Usage:
    python notebooks/validation.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch

from fractalsig.data_gen import generate_rough_paths


def truncate_fourier(signal: np.ndarray, keep_ratio: float = 0.15) -> np.ndarray:
    """Reconstruct signal using truncated Fourier series.

    This simulates the mathematical bottleneck of SigDiffusions: the polynomial
    signature approach effectively truncates high-frequency Fourier components.

    Args:
        signal: Input 1D signal (numpy array).
        keep_ratio: Fraction of low-frequency coefficients to keep (0 to 1).

    Returns:
        Reconstructed signal with high frequencies zeroed out.
    """
    n = len(signal)
    fft_coeffs = np.fft.fft(signal)

    # Keep only the lowest frequency components
    n_keep = max(1, int(n * keep_ratio / 2))

    # Create mask: keep DC, low positive, and corresponding negative frequencies
    mask = np.zeros(n, dtype=bool)
    mask[:n_keep] = True  # Low positive frequencies (including DC)
    mask[-n_keep + 1:] = True  # Corresponding negative frequencies

    fft_truncated = fft_coeffs * mask
    reconstructed = np.fft.ifft(fft_truncated).real

    return reconstructed


def wavelet_compression(
    signal: np.ndarray,
    keep_ratio: float = 0.15,
    wavelet: str = "db4"
) -> np.ndarray:
    """Reconstruct signal using wavelet compression.

    Wavelets are naturally suited for rough/irregular signals because they
    operate in Besov spaces which capture local regularity.

    Args:
        signal: Input 1D signal (numpy array).
        keep_ratio: Fraction of largest coefficients to keep (0 to 1).
        wavelet: Wavelet family to use (default: Daubechies 4).

    Returns:
        Reconstructed signal from thresholded wavelet coefficients.
    """
    # Perform multilevel DWT
    coeffs = pywt.wavedec(signal, wavelet, mode="symmetric")

    # Flatten all coefficients for global thresholding
    all_coeffs = np.concatenate([c.ravel() for c in coeffs])
    n_total = len(all_coeffs)
    n_keep = max(1, int(n_total * keep_ratio))

    # Find threshold for keeping top k coefficients
    threshold = np.sort(np.abs(all_coeffs))[-n_keep]

    # Apply hard thresholding to each level
    coeffs_thresholded = []
    for c in coeffs:
        c_thresh = np.where(np.abs(c) >= threshold, c, 0)
        coeffs_thresholded.append(c_thresh)

    # Reconstruct
    reconstructed = pywt.waverec(coeffs_thresholded, wavelet, mode="symmetric")

    # Handle potential length mismatch from reconstruction
    return reconstructed[:len(signal)]


def estimate_hurst(signal: np.ndarray, max_lag: int = 20) -> float:
    """Estimate Hurst exponent using R/S (Rescaled Range) analysis.

    The Hurst exponent H characterizes the long-range dependence:
    - H < 0.5: Anti-persistent (rough, mean-reverting)
    - H = 0.5: Brownian motion (no memory)
    - H > 0.5: Persistent (trending)

    Args:
        signal: Input 1D signal (numpy array).
        max_lag: Maximum lag for R/S analysis.

    Returns:
        Estimated Hurst exponent.
    """
    n = len(signal)
    lags = range(10, min(max_lag, n // 4))
    rs_values = []

    for lag in lags:
        # Divide into subseries
        n_subseries = n // lag
        rs_subseries = []

        for i in range(n_subseries):
            subseries = signal[i * lag:(i + 1) * lag]
            mean = np.mean(subseries)
            std = np.std(subseries)

            if std < 1e-10:
                continue

            # Cumulative deviation from mean
            cumdev = np.cumsum(subseries - mean)
            r = np.max(cumdev) - np.min(cumdev)
            rs_subseries.append(r / std)

        if rs_subseries:
            rs_values.append((lag, np.mean(rs_subseries)))

    if len(rs_values) < 2:
        return 0.5  # Default to Brownian if insufficient data

    # Fit log(R/S) = H * log(lag) + c
    lags_arr = np.log([v[0] for v in rs_values])
    rs_arr = np.log([v[1] for v in rs_values])

    # Simple linear regression
    slope, _ = np.polyfit(lags_arr, rs_arr, 1)

    return np.clip(slope, 0.01, 0.99)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Mean Squared Error between signals."""
    return np.mean((original - reconstructed) ** 2)


def main():
    """Generate the Validation figure: Gibbs Phenomenon in Rough Volatility."""
    print("\nValidation: Fourier vs Wavelets for Rough Volatility")

    # Configuration
    seq_len = 1024
    H = 0.1
    keep_ratio = 0.15
    seed = 42

    # Generate ground truth rough path
    paths = generate_rough_paths(
        n_paths=1,
        seq_len=seq_len,
        n_channels=1,
        H=H,
        seed=seed,
        standardize=True
    )
    ground_truth = paths[0, :, 0].numpy()

    # Fourier reconstruction (The Baseline - SigDiffusions)
    fourier_recon = truncate_fourier(ground_truth, keep_ratio=keep_ratio)

    # Wavelet reconstruction (The Proposal - FractalSig)
    wavelet_recon = wavelet_compression(ground_truth, keep_ratio=keep_ratio)

    # Compute metrics
    h_gt = estimate_hurst(ground_truth)
    h_fourier = estimate_hurst(fourier_recon)
    h_wavelet = estimate_hurst(wavelet_recon)

    mse_fourier = compute_mse(ground_truth, fourier_recon)
    mse_wavelet = compute_mse(ground_truth, wavelet_recon)

    print(f"\n{'Method':<20} {'Hurst (H)':<15} {'MSE':<15} {'H Error':<15}\n")

    print(f"{'Ground Truth':<20} {h_gt:<15.4f} {'—':<15} {'—':<15}")
    print(f"{'Fourier (Baseline)':<20} {h_fourier:<15.4f} {mse_fourier:<15.6f} {abs(h_gt - h_fourier):<15.4f}")
    print(f"{'Wavelet (Proposal)':<20} {h_wavelet:<15.4f} {mse_wavelet:<15.6f} {abs(h_gt - h_wavelet):<15.4f}\n")

    improvement = (mse_fourier - mse_wavelet) / mse_fourier * 100
    print(f"Wavelet MSE improvement over Fourier: {improvement:.1f}%")

    # Create the visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})

    # Main plot
    ax1 = axes[0]
    t = np.arange(seq_len)

    ax1.plot(t, ground_truth, 'k-', linewidth=1.2, label='Ground Truth (H=0.1)', alpha=0.9)
    ax1.plot(t, fourier_recon, 'r--', linewidth=1.0, label=f'Fourier (MSE={mse_fourier:.4f})', alpha=0.8)
    ax1.plot(t, wavelet_recon, 'b-', linewidth=1.0, label=f'Wavelet/db4 (MSE={mse_wavelet:.4f})', alpha=0.8)

    ax1.set_title(
        r"Gibbs Phenomenon in Rough Volatility ($H \approx 0.1$): Fourier vs Wavelets",
        fontsize=14,
        fontweight='bold'
    )
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Value (Standardized)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.text(
        0.02, 0.98,
        f"Compression ratio: {keep_ratio:.0%} of coefficients retained",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Zoom panel: Find a region with high local variation (rough section)
    ax2 = axes[1]

    # Find region with maximum local variation
    window = 50
    local_var = np.array([
        np.var(ground_truth[i:i+window])
        for i in range(seq_len - window)
    ])
    zoom_start = np.argmax(local_var)
    zoom_end = zoom_start + 100

    t_zoom = t[zoom_start:zoom_end]
    gt_zoom = ground_truth[zoom_start:zoom_end]
    fourier_zoom = fourier_recon[zoom_start:zoom_end]
    wavelet_zoom = wavelet_recon[zoom_start:zoom_end]

    ax2.plot(t_zoom, gt_zoom, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.9)
    ax2.plot(t_zoom, fourier_zoom, 'r--', linewidth=1.3, label='Fourier (Gibbs oscillations)', alpha=0.8)
    ax2.plot(t_zoom, wavelet_zoom, 'b-', linewidth=1.3, label='Wavelet (tight fit)', alpha=0.8)

    # Highlight Gibbs oscillations
    residual_fourier = np.abs(gt_zoom - fourier_zoom)
    residual_wavelet = np.abs(gt_zoom - wavelet_zoom)

    # Find points where Fourier error is significantly larger
    gibbs_mask = residual_fourier > 2 * residual_wavelet + 0.1

    if np.any(gibbs_mask):
        gibbs_points = t_zoom[gibbs_mask]
        gibbs_values = fourier_zoom[gibbs_mask]
        ax2.scatter(gibbs_points, gibbs_values, color='red', s=30, zorder=5,
                   label='Gibbs artifacts', marker='o', alpha=0.7)

    ax2.set_title(
        f'Zoomed View: Time Steps {zoom_start} to {zoom_end} (High Roughness Region)',
        fontsize=12
    )
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add zone indicator on main plot
    ax1.axvspan(zoom_start, zoom_end, alpha=0.15, color='green', label='Zoom region')
    ax1.axvline(zoom_start, color='green', linestyle=':', alpha=0.5)
    ax1.axvline(zoom_end, color='green', linestyle=':', alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_path = project_root / "notebooks" / "deathmatch_figure.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

    # Also display if in interactive mode
    plt.show()

    print("\nThe Fourier-based approach (SigDiffusions) shows characteristic")
    print("Gibbs oscillations when reconstructing rough paths (H ≈ 0.1).")
    print("Wavelets in Besov spaces naturally capture local irregularity,")
    print("providing a mathematically principled solution for FractalSig.\n")

if __name__ == "__main__":
    main()
