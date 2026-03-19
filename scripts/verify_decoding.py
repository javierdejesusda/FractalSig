import torch
import numpy as np
import json
from pathlib import Path
from fractalsig.decoder import FractalDecoder
import pywt

# Setup paths matching notebook
project_root = Path(".").resolve()
checkpoint_dir = project_root / "checkpoints" / "fractal_production"
ckpt_path = checkpoint_dir / "fractal_decoder_best.pth"
stats_path = checkpoint_dir / "normalization_stats.json"
sig_path = project_root / "SigDiffusions" / "data" / "generated_sigs" / "fractal_production.npy"

print(f"Checking paths...")
print(f"Checkpoint: {ckpt_path.exists()}")
print(f"Stats: {stats_path.exists()}")
print(f"Signatures: {sig_path.exists()}")

if not all([ckpt_path.exists(), stats_path.exists(), sig_path.exists()]):
    print("Missing files. Cannot verify.")
    exit(1)

# Load data
with open(stats_path) as f:
    stats_data = json.load(f)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
jax_sigs = np.load(sig_path)

print(f"JAX Sigs Shape: {jax_sigs.shape}")

# Model Init
model = FractalDecoder(
    input_dim=checkpoint["config"]["input_dim"],
    hidden_dim=checkpoint["config"]["hidden_dim"],
    output_seq_len=checkpoint["config"]["seq_len"],
    out_channels=checkpoint["config"]["n_channels"],
    wavelet=checkpoint["config"]["wavelet"],
    level=checkpoint["config"]["level"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Decoding Logic (Snippet from patched notebook)
sigs_t = torch.tensor(jax_sigs, dtype=torch.float32)
ls_mean = torch.tensor(stats_data['logsig_mean'])
ls_std = torch.tensor(stats_data['logsig_std'])

if sigs_t.shape[1] < ls_mean.shape[0]:
    padding = torch.zeros(sigs_t.shape[0], ls_mean.shape[0] - sigs_t.shape[1])
    sigs_t = torch.cat([sigs_t, padding], dim=1)
elif sigs_t.shape[1] > ls_mean.shape[0]:
    sigs_t = sigs_t[:, :ls_mean.shape[0]]

sigs_norm = (sigs_t - ls_mean) / (ls_std + 1e-8)
coeff_mean = np.array(stats_data['coeff_mean'])
coeff_std = np.array(stats_data['coeff_std'])

print(f"Decoding {len(jax_sigs)} signatures...")
reconstructed = []
with torch.no_grad():
    batch = sigs_norm[:10] # Just check a small batch
    pred_norm_flat = model.mlp(batch).numpy()
    pred_denom = pred_norm_flat * coeff_std + coeff_mean
    pred_denom = torch.from_numpy(pred_denom).float()
    coeffs_list_torch = model._unflatten_coefficients(pred_denom)
    coeffs_list_np = [c.numpy() for c in coeffs_list_torch]
    
    for b in range(pred_norm_flat.shape[0]):
        sample_coeffs = [c[b] for c in coeffs_list_np]
        rec = pywt.waverec(sample_coeffs, model.wavelet)
        reconstructed.append(rec[:checkpoint["config"]["seq_len"]])

reconstructed = np.array(reconstructed)
print(f"Reconstructed sample shape: {reconstructed.shape}")
print("Verification successful!")
