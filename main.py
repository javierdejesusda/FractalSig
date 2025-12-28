#!/usr/bin/env python3
"""
FractalSig: The Pipeline

Orchestrator for Rough Volatility Generation.
Unifies JAX Signature Diffusion with PyTorch FractalDecoder.

Modes:
- auto: Run the entire pipeline (intelligent checkpoint detection)
- gen_data: Generate physical training data (.npy)
- train_decoder: Train PyTorch FractalDecoder
- train_jax: Train JAX Signature Diffusion
- sample: Sample JAX signatures -> Decode to rough paths

Hardware Profiles:
- laptop: Optimized for RTX 4070 (8GB VRAM)
- cluster: High performance (A100, V100, etc.)

Usage:
    # Laptop - Full automatic pipeline
    python main.py +profile=laptop mode=auto
    
    # Cluster - Force full retrain
    python main.py +profile=cluster mode=auto force_retrain=true
    
    # Individual steps
    python main.py +profile=laptop mode=gen_data
    python main.py +profile=laptop mode=train_decoder
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Constants
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
SIGDIFFUSIONS_DIR: Path = PROJECT_ROOT / "SigDiffusions"
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Logger (configured by Hydra)
log = logging.getLogger(__name__)


# Utility Functions
def setup_directories() -> None:
    """Create all required directories for the pipeline."""
    dirs: List[Path] = [
        DATA_DIR,
        RESULTS_DIR,
        CHECKPOINTS_DIR,
        SIGDIFFUSIONS_DIR / "data",
        SIGDIFFUSIONS_DIR / "data" / "real_paths",
        SIGDIFFUSIONS_DIR / "data" / "real_sigs",
        SIGDIFFUSIONS_DIR / "data" / "generated_sigs",
        SIGDIFFUSIONS_DIR / "data" / "generated_paths",
        SIGDIFFUSIONS_DIR / "model_checkpoints",
        SIGDIFFUSIONS_DIR / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Detect available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.debug(f"Device: {device}")
    return device


def cleanup_memory() -> None:
    """Aggressive memory cleanup between pipeline steps."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.debug("Memory cleanup completed")


def run_subprocess_checked(
    cmd: List[str], 
    cwd: Path, 
    env: Dict[str, str], 
    desc: str
) -> None:
    """Run a subprocess with error handling and debug logging."""
    log.info(f"Starting: {desc}")
    
    # Capture output to keep console clean, unless debug/error
    result = subprocess.run(
        cmd, 
        cwd=str(cwd), 
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        log.error(f"FATAL: {desc} failed (exit code {result.returncode})")
        log.error(f"STDOUT:\n{result.stdout}")
        log.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Subprocess failed: {desc}")
    else:
        # Log success output as DEBUG so it's hidden by default but available
        if result.stdout:
            log.debug(f"[{desc} STDOUT]\n{result.stdout}")
        if result.stderr:
            log.debug(f"[{desc} STDERR]\n{result.stderr}")
        log.info(f"Completed: {desc}")


# JAX Configuration Injection
def inject_jax_config(cfg: DictConfig) -> Path:
    """
    Dynamically generate SigDiffusions/config/fractal.yaml.
    
    This is for dimension alignment between JAX and PyTorch.
    We enforce:
    - augmentations: "time"
    - input_channels: 1
    - sig_depth: {cfg.pipeline.sig_depth}
    
    Returns:
        Path to the generated config file.
    """
    config_path = SIGDIFFUSIONS_DIR / "config" / "fractal.yaml"
    
    # Compute signature dimension for the given depth
    # For time-augmented paths (2D), log-sig dimension = siglength(2, depth)
    # Using formula from iisignature: siglength(d, m) = (d^(m+1) - 1) / (d - 1) - 1
    d = 2  # Time + 1 channel
    depth = cfg.pipeline.sig_depth
    sig_dim = int((d ** (depth + 1) - 1) / (d - 1)) - 1
    
    log.info(f"Injecting JAX config: sig_depth={depth}, expected_sig_dim={sig_dim}")
    
    # Build the configuration
    jax_config = {
        "seed": 42,
        
        "logging_folders": {
            "real_paths": "./data/real_paths/",
            "real_sigs": "./data/real_sigs/",
            "generated_sigs": "./data/generated_sigs/",
            "generated_paths": "./data/generated_paths/",
            "model_checkpoints": "./model_checkpoints/",
        },
        
        "dataset": {
            "data_path": "data/rough_volatility.npy",
            "preprocessing_fn": "data_loading_utils.load_numpy_data",
            "seq_len": cfg.gen_data.seq_len,
            "dim": 2,  # Time augmentation (time + value)
            "scaler": None,
            "shuffle": True,
            "sig_depth": cfg.pipeline.sig_depth,  # ENFORCED
            "by_channel": False,
            "mirror_augmentation": False,
            "test_set_size": 100,  # Reduced for memory
        },
        
        "model": {
            # Compact model for 8GB GPU with sig_depth=7
            # sig_dim=254 requires smaller hidden sizes
            "hidden_size": 32,
            "hidden_size_multiplier": 2,
            "num_layers": 2,
            "num_heads": 2,
        },
        
        "training": {
            "num_epochs": 200,
            "batch_size": min(cfg.pipeline.batch_size, 16),  # Cap at 16 for memory
            "print_every": 20,
            "lr": 0.001,
        },
        
        "sampling": {
            "num_steps": 64,
            "sample_size": cfg.sample.n_samples,
            "sample_batch_size": min(25, cfg.sample.n_samples),  # Reduced batch
        },
    }
    
    # Write the config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(f"# AUTO-GENERATED by FractalSig main.py\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# sig_depth: {depth} -> expected dimension: {sig_dim}\n")
        f.write(f"# CRITICAL: Do not edit manually, run main.py to regenerate.\n\n")
        yaml.dump(jax_config, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"Generated JAX config: {config_path}")
    return config_path


# Pipeline Step Check Functions
def check_data_exists(cfg: DictConfig) -> bool:
    """Check if training data exists."""
    data_path = PROJECT_ROOT / cfg.gen_data.output
    exists = data_path.exists()
    log.debug(f"Data check: {data_path} -> {'EXISTS' if exists else 'MISSING'}")
    return exists


def check_decoder_exists(cfg: DictConfig) -> bool:
    """Check if trained decoder checkpoint exists."""
    ckpt_path = CHECKPOINTS_DIR / cfg.experiment / "fractal_decoder_best.pth"
    exists = ckpt_path.exists()
    log.debug(f"Decoder check: {ckpt_path} -> {'EXISTS' if exists else 'MISSING'}")
    return exists


def check_jax_trained(cfg: DictConfig) -> bool:
    """Check if JAX model has been trained."""
    ckpt_path = SIGDIFFUSIONS_DIR / "model_checkpoints" / f"{cfg.experiment}.pkl"
    exists = ckpt_path.exists()
    log.debug(f"JAX check: {ckpt_path} -> {'EXISTS' if exists else 'MISSING'}")
    return exists


# Core Pipeline Steps
def cmd_gen_data(cfg: DictConfig) -> None:
    """Generate physical training data (rough volatility paths)."""
    log.info("\nStep: Data Generation")
    
    from fractalsig.data_gen import generate_rough_paths
    
    start_time = time.perf_counter()
    n_samples = cfg.gen_data.n_samples
    output_path = PROJECT_ROOT / cfg.gen_data.output
    
    # Generate in chunks for memory efficiency
    chunk_size = 1000
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunks: List[torch.Tensor] = []
    
    pbar = tqdm(total=n_samples, desc="Generating fBM Paths", unit="path")
    for i in range(n_chunks):
        current_batch = min(chunk_size, n_samples - (i * chunk_size))
        paths = generate_rough_paths(
            n_paths=current_batch,
            seq_len=cfg.gen_data.seq_len,
            n_channels=1,
            H=cfg.gen_data.H,
            seed=cfg.gen_data.seed + i,
            standardize=True
        )
        
        # Time augmentation (prepend time channel)
        time_ch = torch.linspace(0, 1, cfg.gen_data.seq_len)
        time_ch = time_ch.view(1, -1, 1).expand(current_batch, -1, -1)
        data = torch.cat([time_ch, paths], dim=-1)
        chunks.append(data)
        pbar.update(current_batch)
    pbar.close()
    
    # Concatenate and save
    final_data = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, final_data)
    
    # Also copy to SigDiffusions/data for JAX
    jax_data_path = SIGDIFFUSIONS_DIR / "data" / "rough_volatility.npy"
    np.save(jax_data_path, final_data)
    
    duration = time.perf_counter() - start_time
    log.info(f"Generated {n_samples} paths ({final_data.shape}) in {duration:.2f}s")
    log.info(f"Saved to: {output_path}")
    log.info(f"Copied to: {jax_data_path}")
    
    cleanup_memory()


def cmd_train_decoder(cfg: DictConfig) -> None:
    """Train PyTorch FractalDecoder."""
    log.info("\nStep: PyTorch Decoder Training")
    
    from fractalsig.train_decoder import train as train_decoder_fn
    
    start_time = time.perf_counter()
    checkpoint_dir = CHECKPOINTS_DIR / cfg.experiment
    
    log.info(f"Config: sig_depth={cfg.pipeline.sig_depth}, "
             f"hidden_dim={cfg.pipeline.hidden_dim}, "
             f"batch_size={cfg.pipeline.batch_size}")
    
    train_decoder_fn(
        n_samples=cfg.train_decoder.n_samples,
        sig_depth=cfg.pipeline.sig_depth,
        hidden_dim=cfg.pipeline.hidden_dim,
        epochs=cfg.pipeline.decoder_epochs,
        batch_size=cfg.pipeline.batch_size,
        H=cfg.train_decoder.H,
        checkpoint_dir=str(checkpoint_dir),
        device=get_device(),
    )
    
    duration = time.perf_counter() - start_time
    log.info(f"Decoder training completed in {duration:.2f}s")
    log.info(f"Checkpoint saved to: {checkpoint_dir}")
    
    cleanup_memory()


def cmd_train_jax(cfg: DictConfig) -> None:
    """Train JAX Signature Diffusion model."""
    log.info("\nStep: JAX Training")
    
    start_time = time.perf_counter()
    
    # Inject JAX config (CRITICAL for dimension alignment)
    config_path = inject_jax_config(cfg)
    
    # Prepare environment
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.jax.memory_fraction)
    
    # Compute Signatures
    log.info("Computing signatures...")
    run_subprocess_checked(
        [sys.executable, "main.py", "compute-sigs", cfg.experiment, str(config_path.relative_to(SIGDIFFUSIONS_DIR))],
        cwd=SIGDIFFUSIONS_DIR,
        env=env,
        desc="JAX Signature Computation"
    )
    
    cleanup_memory()
    
    # Train Model
    log.info("Training diffusion model...")
    run_subprocess_checked(
        [sys.executable, "main.py", "train", cfg.experiment, str(config_path.relative_to(SIGDIFFUSIONS_DIR))],
        cwd=SIGDIFFUSIONS_DIR,
        env=env,
        desc="JAX Training"
    )
    
    duration = time.perf_counter() - start_time
    log.info(f"JAX training completed in {duration:.2f}s")
    
    cleanup_memory()


def cmd_sample(cfg: DictConfig) -> None:
    """Sample from JAX model and decode with PyTorch."""
    log.info("\nStep: Sampling & Decoding")
    
    import matplotlib.pyplot as plt
    from fractalsig.decoder import FractalDecoder
    
    start_time = time.perf_counter()
    
    # Inject JAX config to ensure consistency
    config_path = inject_jax_config(cfg)
    
    # JAX Sampling
    log.info("Sampling from JAX diffusion model...")
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.jax.memory_fraction)
    
    run_subprocess_checked(
        [sys.executable, "main.py", "sample", cfg.experiment, str(config_path.relative_to(SIGDIFFUSIONS_DIR))],
        cwd=SIGDIFFUSIONS_DIR,
        env=env,
        desc="JAX Sampling"
    )
    
    cleanup_memory()
    
    # PyTorch Decoding
    log.info("Decoding signatures with FractalDecoder...")
    
    # Load generated signatures
    sig_path = SIGDIFFUSIONS_DIR / "data" / "generated_sigs" / f"{cfg.experiment}.npy"
    if not sig_path.exists():
        raise FileNotFoundError(f"Generated signatures not found: {sig_path}")
    
    jax_sigs = np.load(sig_path)
    log.info(f"Loaded {jax_sigs.shape[0]} signatures (dim={jax_sigs.shape[1]})")
    
    # Load decoder
    device = get_device()
    checkpoint_dir = CHECKPOINTS_DIR / cfg.experiment
    ckpt_path = checkpoint_dir / "fractal_decoder_best.pth"
    stats_path = checkpoint_dir / "normalization_stats.json"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    with open(stats_path) as f:
        stats = json.load(f)
    
    # Initialize model
    model = FractalDecoder(
        input_dim=checkpoint["config"]["input_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        output_seq_len=checkpoint["config"]["seq_len"],
        out_channels=checkpoint["config"]["n_channels"],
        wavelet=checkpoint["config"]["wavelet"],
        level=checkpoint["config"]["level"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Prepare signatures
    sigs_t = torch.from_numpy(jax_sigs).float().to(device)
    logsig_mean = torch.tensor(stats["logsig_mean"]).float().to(device)
    logsig_std = torch.tensor(stats["logsig_std"]).float().to(device)
    
    # Handle dimension mismatch
    if sigs_t.shape[1] != logsig_mean.shape[0]:
        log.warning(f"Dimension mismatch: JAX={sigs_t.shape[1]}, Decoder={logsig_mean.shape[0]}")
        target_dim = logsig_mean.shape[0]
        if sigs_t.shape[1] > target_dim:
            log.warning(f"Truncating signatures to {target_dim} dimensions")
            sigs_t = sigs_t[:, :target_dim]
        else:
            log.warning(f"Padding signatures to {target_dim} dimensions")
            padding = torch.zeros(sigs_t.shape[0], target_dim - sigs_t.shape[1]).to(device)
            sigs_t = torch.cat([sigs_t, padding], dim=1)
    else:
        log.info("✓ Dimensions aligned perfectly!")
    
    # Normalize and decode
    sigs_norm = (sigs_t - logsig_mean) / (logsig_std + 1e-8)
    
    with torch.no_grad():
        decoded_paths = model(sigs_norm).cpu().numpy()
    
    # Validate roughness
    roughness = np.diff(decoded_paths[:, :, 0], axis=1).std()
    log.info(f"Roughness Metric (Increment Std): {roughness:.4f}")
    if roughness > 0.8:
        log.info("✓ Valid Roughness - Paths exhibit rough volatility characteristics")
    else:
        log.warning("⚠ Result may be too smooth")
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / f"{cfg.experiment}_final_paths.npy", decoded_paths)
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(decoded_paths):
            ax.plot(decoded_paths[i, :, 0], "b-", lw=0.6, alpha=0.8)
            ax.set_title(f"Generated Path {i+1}")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Value")
    
    plt.suptitle(
        f"FractalSig: Generated Rough Volatility Paths\n"
        f"Experiment: {cfg.experiment} | Roughness: {roughness:.4f}",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cfg.experiment}_final.png", dpi=150)
    plt.close()
    
    duration = time.perf_counter() - start_time
    log.info(f"Sampling & Decoding completed in {duration:.2f}s")
    log.info(f"Results saved to: {RESULTS_DIR}")
    
    cleanup_memory()


def cmd_auto(cfg: DictConfig) -> None:
    """
    The full pipeline.
    
    Intelligent pipeline that checks for existing artifacts
    and only runs necessary steps.
    """
    log.info("\nAUTO MODE: Intelligent Pipeline")
    
    force = cfg.force_retrain
    if force:
        log.info("force_retrain=True: Ignoring all checkpoints")
    
    steps_run = []
    
    # Check/Generate Data
    if force or not check_data_exists(cfg):
        log.info("[AUTO] Data missing -> Running gen_data")
        cmd_gen_data(cfg)
        steps_run.append("gen_data")
    else:
        log.info("[AUTO] Data exists -> Skipping gen_data")
    
    # Check/Train Decoder
    if force or not check_decoder_exists(cfg):
        log.info("[AUTO] Decoder missing -> Running train_decoder")
        cmd_train_decoder(cfg)
        steps_run.append("train_decoder")
    else:
        log.info("[AUTO] Decoder exists -> Skipping train_decoder")
    
    # Check/Train JAX
    if force or not check_jax_trained(cfg):
        log.info("[AUTO] JAX model missing -> Running train_jax")
        cmd_train_jax(cfg)
        steps_run.append("train_jax")
    else:
        log.info("[AUTO] JAX model exists -> Skipping train_jax")
    
    # Always sample at the end
    log.info("[AUTO] Running final sampling & decoding")
    cmd_sample(cfg)
    steps_run.append("sample")
    
    log.info(f"AUTO MODE COMPLETE: Executed steps: {', '.join(steps_run)}")


# Main Entry Point
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """FractalSig Pipeline Dispatcher."""
    
    # Change back to the original working directory
    # (Hydra changes CWD to outputs folder)
    os.chdir(PROJECT_ROOT)
    
    setup_directories()
    mode = cfg.mode.lower()
    
    log.info(f"FractalSig Pipeline | Mode: {mode.upper()} | Experiment: {cfg.experiment}")
    log.debug(f"Profile Settings: batch_size={cfg.pipeline.batch_size}, "
             f"hidden_dim={cfg.pipeline.hidden_dim}, "
             f"sig_depth={cfg.pipeline.sig_depth}")
    
    try:
        if mode == "auto":
            cmd_auto(cfg)
        elif mode == "gen_data":
            cmd_gen_data(cfg)
        elif mode == "train_decoder":
            cmd_train_decoder(cfg)
        elif mode == "train_jax":
            cmd_train_jax(cfg)
        elif mode == "sample":
            cmd_sample(cfg)
        else:
            log.error(f"Unknown mode: {mode}")
            log.info("Available modes: auto, gen_data, train_decoder, train_jax, sample")
            sys.exit(1)
            
        log.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.exception(f"FATAL ERROR in {mode}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
