"""Learned Wavelet Decoder for Rough Path Generation.

This module implements the FractalDecoder - an nn.Module that maps a compressed
Log-Signature embedding to a full wavelet coefficient tree, enabling differentiable
reconstruction of rough paths.

The key insight: Wavelets in Besov spaces naturally capture the multi-scale,
irregular structure of rough volatility (H ≈ 0.1), making them the ideal basis
for "hallucinating" high-frequency details from low-dimensional signatures.

Why Shape Inference?
-------------------
Wavelet decomposition produces coefficient arrays of varying sizes at each level.
Rather than hardcoding these shapes (which would break for different sequence
lengths or wavelet families), we perform dynamic shape inference at initialization
using pywt. This ensures the MLP output dimension exactly matches the flattened
wavelet coefficient structure, regardless of configuration.
"""


import numpy as np

# ptwt provides differentiable wavelet transforms for PyTorch
import ptwt
import pywt
import torch
import torch.nn as nn


def _safe_n_heads(hidden_dim: int, preferred: int) -> int:
    """Pick the largest n_heads up to ``preferred`` that divides ``hidden_dim``."""
    for h in range(min(preferred, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


class _MLPAttnBackbone(nn.Module):
    """MLP head with one self-attention block over projected token groups.

    The input vector is projected to a small bag of ``n_tokens`` tokens of
    width ``hidden_dim``, attended over once, then flattened and projected
    to the flat coefficient vector. This adds explicit cross-scale mixing
    without ballooning parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        n_tokens: int = 4,
    ) -> None:
        super().__init__()
        n_heads = _safe_n_heads(hidden_dim, preferred=4)
        self.n_tokens = n_tokens
        self.token_dim = hidden_dim

        in_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, n_layers - 3)):
            in_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ]
        self.in_proj = nn.Sequential(*in_layers)
        self.tokenize = nn.Linear(hidden_dim, n_tokens * hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(n_tokens * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        tok = self.tokenize(h).view(-1, self.n_tokens, self.token_dim)
        attn_out, _ = self.attn(tok, tok, tok, need_weights=False)
        h2 = self.norm(tok + attn_out)
        return self.out_proj(h2.flatten(1))


class _TransformerBackbone(nn.Module):
    """Single TransformerEncoder layer over projected token groups.

    Sized to stay within roughly 4x the MLP backbone's parameter budget at
    typical settings: one encoder layer, ``dim_feedforward = hidden_dim``,
    and ``n_tokens = 4`` projected tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        n_tokens: int = 4,
    ) -> None:
        super().__init__()
        n_heads = _safe_n_heads(hidden_dim, preferred=2)
        self.n_tokens = n_tokens
        self.token_dim = hidden_dim
        self.tokenize = nn.Linear(input_dim, n_tokens * hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(n_tokens * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.tokenize(x).view(-1, self.n_tokens, self.token_dim) + self.pos
        h = self.encoder(tok)
        return self.out_proj(h.flatten(1))


class FractalDecoder(nn.Module):
    """Learned decoder mapping Log-Signatures to Wavelet coefficient trees.

    This module serves as the "hallucination engine" - it learns to predict
    high-frequency wavelet details from the geometric summary (signature) of
    a rough path. The architecture uses an MLP backbone followed by
    differentiable wavelet reconstruction via ptwt.

    Architecture:
        LogSig -> MLP -> Flat Coefficients -> Reshape -> [cA, cD_n, ..., cD_1] -> IDWT -> Path

    Attributes:
        input_dim: Dimension of input log-signature.
        hidden_dim: Hidden dimension for MLP layers.
        output_seq_len: Target sequence length for reconstructed paths.
        out_channels: Number of output channels.
        wavelet: Wavelet family (e.g., 'db4').
        level: Decomposition level (computed automatically if None).
        coeff_shapes: List of shapes for each coefficient array.
        coeff_slices: Slicing indices for unflattening MLP output.
        total_coeff_dim: Total flattened dimension of all coefficients.

    Example:
        >>> decoder = FractalDecoder(input_dim=64, hidden_dim=256, output_seq_len=256)
        >>> log_sig = torch.randn(32, 64)  # batch of 32
        >>> paths = decoder(log_sig)
        >>> paths.shape
        torch.Size([32, 256, 1])
    """

    SUPPORTED_ARCHS = ("mlp", "mlp_attn", "transformer")

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_seq_len: int,
        out_channels: int = 1,
        wavelet: str = "db4",
        level: int | None = None,
        n_layers: int = 4,
        dropout: float = 0.1,
        arch: str = "mlp",
    ) -> None:
        """Initialize the FractalDecoder.

        Args:
            input_dim: Dimension of the input log-signature vector.
            hidden_dim: Hidden dimension for backbone layers.
            output_seq_len: Target length of reconstructed sequences.
            out_channels: Number of output channels (default: 1).
            wavelet: Wavelet family to use (default: 'db4' - Daubechies 4).
            level: Wavelet decomposition level. If None, computed automatically
                based on sequence length.
            n_layers: Number of MLP layers (default: 4 for sufficient non-linearity).
            dropout: Dropout rate for regularization (default: 0.1).
            arch: Backbone variant — ``"mlp"`` (default), ``"mlp_attn"``
                (MLP plus one self-attention block over projected token
                groups), or ``"transformer"`` (single TransformerEncoder
                layer over projected tokens). All variants expose the same
                ``(input_dim,) -> (total_coeff_dim,)`` interface as ``self.mlp``
                so the rest of the pipeline is unchanged. The transformer
                variant is sized to stay within roughly 4x the MLP
                backbone's parameter budget for fair ablation.

        Raises:
            ValueError: If ``arch`` is not one of ``SUPPORTED_ARCHS``.
        """
        super().__init__()

        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(
                f"arch must be one of {self.SUPPORTED_ARCHS}, got {arch!r}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len
        self.out_channels = out_channels
        self.wavelet = wavelet
        self.n_layers = n_layers
        self.arch = arch

        if level is None:
            wavelet_obj = pywt.Wavelet(wavelet)
            max_level = pywt.dwt_max_level(output_seq_len, wavelet_obj.dec_len)
            self.level = max(1, min(max_level, 6))
        else:
            self.level = level

        self._infer_coefficient_shapes()

        self.mlp = self._build_backbone(arch, dropout)

    def _infer_coefficient_shapes(self) -> None:
        """Infer wavelet coefficient shapes via dummy decomposition.

        This is crucial for building a flexible decoder that works with
        any sequence length or wavelet family. We run pywt.wavedec on a
        zero array and inspect the resulting coefficient structure.
        """
        # Dummy signal for shape inference
        dummy_signal = np.zeros(self.output_seq_len)
        coeffs = pywt.wavedec(dummy_signal, self.wavelet, level=self.level)

        # Store shapes and compute slicing indices
        self.coeff_shapes: list[tuple[int, ...]] = []
        self.coeff_slices: list[tuple[int, int]] = []

        current_idx = 0
        for coeff in coeffs:
            shape = coeff.shape
            size = int(np.prod(shape))
            self.coeff_shapes.append(shape)
            self.coeff_slices.append((current_idx, current_idx + size))
            current_idx += size

        # Total flattened dimension (per channel)
        self.total_coeff_dim_per_channel = current_idx
        self.total_coeff_dim = current_idx * self.out_channels

        # Register shapes as buffer for device handling (stored but not trained)
        self.register_buffer(
            "_coeff_lengths",
            torch.tensor([s[0] for s in self.coeff_shapes], dtype=torch.long)
        )

    def _build_backbone(self, arch: str, dropout: float) -> nn.Module:
        """Build the chosen backbone mapping ``input_dim`` to ``total_coeff_dim``.

        Args:
            arch: One of ``"mlp"``, ``"mlp_attn"``, ``"transformer"``.
            dropout: Dropout rate forwarded to all sub-modules.

        Returns:
            ``nn.Module`` whose forward maps ``(B, input_dim)`` to
            ``(B, total_coeff_dim)``.
        """
        if arch == "mlp":
            return self._build_mlp(dropout)
        if arch == "mlp_attn":
            return _MLPAttnBackbone(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.total_coeff_dim,
                n_layers=self.n_layers,
                dropout=dropout,
            )
        if arch == "transformer":
            return _TransformerBackbone(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.total_coeff_dim,
                dropout=dropout,
            )
        raise ValueError(f"unknown arch {arch!r}")

    def _build_mlp(self, dropout: float) -> nn.Sequential:
        """Build the MLP backbone for learning Signature → Wavelet mapping.

        Args:
            dropout: Dropout rate for regularization.

        Returns:
            Sequential MLP module.
        """
        layers: list[nn.Module] = []

        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.Dropout(dropout))

        for _ in range(self.n_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(self.hidden_dim, self.total_coeff_dim))

        return nn.Sequential(*layers)

    def _unflatten_coefficients(
        self,
        flat_coeffs: torch.Tensor
    ) -> list[torch.Tensor]:
        """Reshape flat MLP output into wavelet coefficient structure.

        Args:
            flat_coeffs: Tensor of shape (batch, total_coeff_dim_per_channel).

        Returns:
            List of coefficient tensors [cA, cD_n, cD_{n-1}, ..., cD_1]
            where cA is the approximation and cD_i are detail coefficients.
        """
        batch_size = flat_coeffs.shape[0]
        coeffs = []

        for (start, end), shape in zip(self.coeff_slices, self.coeff_shapes, strict=False):
            # Extract and reshape coefficient
            coeff = flat_coeffs[:, start:end]
            # ptwt expects (batch, coeff_len) for 1D wavelets
            coeff = coeff.view(batch_size, shape[0])
            coeffs.append(coeff)

        return coeffs

    def forward(self, log_signature: torch.Tensor) -> torch.Tensor:
        """Forward pass: Log-Signature → Reconstructed Path.

        The MLP runs once and projects to all channels jointly. Its output
        is reshaped to (batch, out_channels, total_coeff_dim_per_channel)
        so the linear head can learn cross-channel correlations rather than
        treating channels as independent slices of an unstructured vector.

        Args:
            log_signature: Input tensor of shape (batch, input_dim) containing
                the log-signature embeddings.

        Returns:
            Reconstructed paths of shape (batch, output_seq_len, out_channels).
        """
        wavelet = pywt.Wavelet(self.wavelet)
        batch_size = log_signature.shape[0]

        full_output = self.mlp(log_signature)
        per_channel = full_output.view(
            batch_size, self.out_channels, self.total_coeff_dim_per_channel
        )

        outputs = []
        for c in range(self.out_channels):
            flat_coeffs = per_channel[:, c, :]
            coeffs = self._unflatten_coefficients(flat_coeffs)
            reconstructed = ptwt.waverec(coeffs, wavelet)
            reconstructed = reconstructed[:, :self.output_seq_len]
            outputs.append(reconstructed)

        return torch.stack(outputs, dim=-1)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        """Extra representation for print(model)."""
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"output_seq_len={self.output_seq_len}, out_channels={self.out_channels}, "
            f"wavelet='{self.wavelet}', level={self.level}, "
            f"total_coeff_dim={self.total_coeff_dim}"
        )


if __name__ == "__main__":
    # Sanity check
    print("FractalDecoder Sanity Check")

    # Configuration
    batch_size = 16
    input_dim = 64  # Typical log-signature dimension
    hidden_dim = 256
    seq_len = 256
    n_channels = 1

    # Instantiate model
    model = FractalDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_seq_len=seq_len,
        out_channels=n_channels,
        wavelet="db4",
    )

    print(f"\nModel:\n{model}")
    print(f"\nWavelet decomposition level: {model.level}")
    print(f"Coefficient shapes: {model.coeff_shapes}")
    print(f"Total coefficient dimension: {model.total_coeff_dim}")

    # Forward pass with random input
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    # Verify output shape
    expected_shape = (batch_size, seq_len, n_channels)
    assert output.shape == expected_shape, (
        f"Shape mismatch, Expected {expected_shape}, got {output.shape}"
    )

    # Verify differentiability
    loss = output.sum()
    loss.backward()

    grad_exists = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters() if p.requires_grad
    )
    assert grad_exists, "Gradients did not flow through the model!"

    # Summary
    total_params = model.get_num_params()
    print(f"Model initialized. Total params: {total_params:,}")
    print(f"Output shape verified: {output.shape}")
    print("Differentiability verified: gradients flow correctly")
