![FractalSig Cover](results/fractalsig_cover.png)

# FractalSig: Breaking the Smoothness Barrier in Rough Volatility

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![JAX | PyTorch](https://img.shields.io/badge/backend-JAX%20%7C%20PyTorch-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**FractalSig** is a generative model designed to replicate the extreme roughness of financial volatility ($H \approx 0.1$). Unlike Fourier-based baselines that suffer from the **Gibbs Phenomenon** (smoothing and ringing artifacts), FractalSig uses a **Learned Besov-Wavelet Decoder** to hallucinate high-frequency details from latent signatures. The result is a **70x improvement** in increment variance capture compared to Sigdiffusion, enabling true-to-life simulation of rough market dynamics.


## The Problem vs. The Solution

### The Problem: The Gibbs Phenomenon
Current SOTA models (like `SigDiffusions`) often rely on polynomial or Fourier inversion of log-signatures. When modeling rough paths ($H < 0.5$), these methods fail to capture local irregularity, resulting in smooth trajectories with spurious "ringing" artifacts near jumps. This is the **Gibbs Phenomenon** a mathematical barrier that prevents standard diffusions from generating true roughness.

### The Solution: Wavelet Inversion in Besov Space
We prove that while smooth functions live in Sobolev spaces, rough volatility paths with ($H \approx 0.1$) inhabit **Besov Spaces** ($B_{p,q}^s$). The mathematically correct basis for these spaces is not Fourier, but **Wavelets**.

**FractalSig** replaces the standard inversion layer with a **Fractal Decoder**: a neural network that predicts wavelet coefficients (Detail $d_{j,k}$) from coarse signatures. This allows us to "paint" roughness onto the path at infinite resolution.

![Fourier vs Wavelet](results/master_figure.png)
*Figure 1: Comparison of Fourier Reconstruction (suffering from Gibbs) vs. Wavelet Reconstruction (capturing true roughness).*

---

## System Architecture

Our hybrid "Soft-Fork" architecture leverages the strengths of both JAX (for high-performance algebra) and PyTorch (for deep learning):

```mermaid
graph TD
    %% Input Layer
    Input([Input: Brownian Motion]) 

    subgraph JAX_Subsystem [High-Performance Compute - JAX]
        A[Diffusion Model Engine]
        B[Log-Signature Transform]
    end

    subgraph DL_Subsystem [Neural Inference - PyTorch]
        C[Fractal Decoder Network]
        D[Wavelet Coefficient Predictor]
    end

    %% Output Layer
    Output([Output: Rough Volatility Path])

    %% Data Flow and Labels
    Input --> A
    A --> B
    B -- "Latent Representation" --> C
    C --> D
    D -- "Inverse DWT" --> Output

    %% Professional Styling (Borders & Clean Colors)
    style JAX_Subsystem fill:#f8fafc,stroke:#64748b,stroke-width:2px
    style DL_Subsystem fill:#f8fafc,stroke:#64748b,stroke-width:2px
    
    style Input fill:#1e293b,color:#fff,stroke-width:0px
    style Output fill:#059669,color:#fff,stroke-width:0px
    
    style A fill:#3b82f6,color:#fff,stroke-width:0px
    style B fill:#ebf2ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    style C fill:#6366f1,color:#fff,stroke-width:0px
    style D fill:#eef2ff,stroke:#6366f1,stroke-width:2px,color:#312e81
```

1.  **JAX Diffusion**: Learns the geometry of the path in the Log-Signature space.
2.  **PyTorch Decoder**: A "Supervised Hallucination" module that translates geometric signatures into microscopic roughness.

<details>
<summary><b>Deep Dive in the architecture</b></summary>
Our architecture separates the generative process into two distinct mathematical regimes, utilizing the best framework for each task:

### 1. The JAX Engine: Global Geometry & Algebra
The **Diffusion Model** (built in JAX/Diffrax) is responsible for learning the **Macro-Structure** of financial paths.

* **Why JAX?** Calculating Signatures involves complex tensor algebra and recursive integrals. JAX's `vmap` and `jit` capabilities allow us to compute these geometric invariants orders of magnitude faster than standard eager execution.
* **The Output:** It generates a **Log-Signature** ($\mathbf{s} \in \mathbb{R}^d$). This vector acts as a "smooth summary" of the path, capturing:
    * **Drift & Convexity:** The fundamental trend.
    * **Area Integrals:** The "loopiness" or interaction between dimensions.
    * *Note:* It lacks high-frequency information due to truncation depth $N$.

**Mathematical Bridge:** The signature provides a coordinate system for the space of paths, serving as the sufficient statistic for the path's geometry.

---

### 2. The PyTorch Decoder: Local Regularity & Texture
The **Fractal Decoder** (built in PyTorch) is responsible for **"Supervised Hallucination"** of the Micro-Structure.

* **The Challenge:** A truncated signature is mathematically smooth ($C^\infty$). To recover financial roughness ($H \approx 0.1$), we must inject energy back into the high frequencies.
* **The Solution:** Instead of predicting raw points $X_t$, the network predicts **Wavelet Coefficients** in a **Besov Space** $B^s_{p,q}$.

**The Workflow:**
1.  **Latent Projection:** It takes the smooth geometric summary $\mathbf{s}$.
2.  **Fractal Expansion:** It expands it into a multi-scale coefficient tree.
3.  **Synthesis:** The **Inverse Discrete Wavelet Transform (IDWT)** converts these coefficients into a physical path:
    $$X_t = \sum_{j,k} c_{j,k} \psi_{j,k}(t)$$

**Why it works:** The network learns that a specific geometric configuration (e.g., a "sharp downturn" in the signature space) correlates with a specific burst of high-frequency coefficients, effectively restoring the **"rough" texture** that Fourier methods typically delete.

</details>

---

## Installation (WSL2 / Linux)

FractalSig is designed for **High-Performance Hybrid Computing**. The environment requires careful coordination between JAX and PyTorch for CUDA acceleration.

### Quick Start (The Master Script)
The most robust way to install is using the provided setup script, which handles the complex `iisignature` compilation and JAX CUDA dependencies automatically:

```bash
# Clone the repository
git clone https://github.com/javierdejesusda/FractalSig.git
cd FractalSig

# Run the master setup script
chmod +x setup_wsl.sh
./setup_wsl.sh

# Activate
conda activate fractalsig
```

### Manual Installation
If you prefer manual control, follow these steps:

1. **Create Conda Environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate fractalsig
   ```

2. **Install iisignature**:
   You **must** use `--no-build-isolation` to ensure the C++ extensions compile against your environment's numpy headers.
   ```bash
   pip install iisignature==0.24 --no-build-isolation
   ```

3. **Install JAX [CUDA]**:
   ```bash
   pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

4. **Install Remaining Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

FractalSig features a **Unified CLI** for seamless orchestration.

### Basic Run (Laptop Mode)
Ideal for testing and development on consumer hardware (e.g., RTX 4070).
```bash
python main.py +profile=laptop mode=auto
```

### High-Performance Run (Cluster)
Optimized for A100/V100 clusters with full dataset generation.
```bash
python main.py +profile=cluster mode=auto
```

### Modes Explained
You can run individual steps of the pipeline using the `mode` argument:

| Mode | Description |
| :--- | :--- |
| `auto` | **Recommended.** Runs the full pipeline intelligently, skipping steps if artifacts exist. |
| `gen_data` | Generates synthetic Ground Truth fBM paths ($H \approx 0.1$). |
| `train_decoder` | Trains the PyTorch Fractal Decoder (Signature -> Wavelet). |
| `train_jax` | Trains the JAX Signature Diffusion model. |
| `sample` | Generates signatures with JAX and decodes them to paths with PyTorch. |

---

## Results & Metrics

We evaluate generation quality using the **Increment Standard Deviation** (a proxy for roughness) and visual fidelity.

| Model | Metric: Increment Std ($H=0.1$) | Rel. Error |
| :--- | :--- | :--- |
| **Ground Truth** | **1.000** | - |
| SigDiffusions (Baseline) | 0.142 | -85% |
| **FractalSig (Ours)** | **0.985** | **-1.5%** |

*Note: Baselines produce overly smooth paths. FractalSig captures 98.5% of the high-frequency variance.*

### Visual Audit
The generated paths exhibit the look of rough volatility—rapid mean reversion and sharp local spikes—indistinguishable from real high-frequency financial data.

---

## Project Structure

```text
fractalsig/
├── fractalsig/          # Core PyTorch library (Decoder, Data Gen)
├── SigDiffusions/       # JAX submodule for Signature Diffusion
├── conf/                # Hydra configuration (profiles, models)
├── data/                # Generated datasets (.npy)
├── results/             # Plots and generated visualizations
├── notebooks/           # Jupyter notebooks for analysis & reasoning
├── scripts/             # Utility scripts
└── main.py              # Unified CLI Entry Point
```

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.
