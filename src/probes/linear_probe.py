"""Probe architectures for probing ViT hidden states."""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Simple linear probe: one linear layer per patch token.

    Input: (B, 196, 768) patch-level features
    Output: (B, 196) logits (one per patch)
    Parameters: 768 + 1 = 769
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, hidden_dim) or (B, hidden_dim) for single patch
        Returns:
            logits: (B, num_patches) or (B,)
        """
        logits = self.linear(x).squeeze(-1)
        return logits


class MLPProbe(nn.Module):
    """Two-layer MLP probe with ReLU and dropout.

    Input: (B, 196, 768) patch-level features
    Output: (B, 196) logits
    Parameters: 768*256 + 256 + 256*1 + 1 ≈ 197K
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x).squeeze(-1)
        return logits


class ConvProbe(nn.Module):
    """Convolutional probe that operates on the 14x14 spatial grid.

    Reshapes patch tokens to (B, 768, 14, 14), applies Conv2d layers.
    This captures spatial relationships between neighboring patches.

    Input: (B, 196, 768) patch-level features
    Output: (B, 196) logits
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 196, 768)
        Returns:
            logits: (B, 196)
        """
        B = x.shape[0]
        # Reshape to spatial grid: (B, 196, 768) -> (B, 768, 14, 14)
        x = x.transpose(1, 2).reshape(B, self.input_dim, self.grid_size, self.grid_size)
        # Apply convolutions: (B, 768, 14, 14) -> (B, 1, 14, 14)
        x = self.conv(x)
        # Flatten back: (B, 1, 14, 14) -> (B, 196)
        x = x.reshape(B, -1)
        return x


def get_probe(probe_type: str, input_dim: int = 768, **kwargs) -> nn.Module:
    """Factory function to create probe by type name."""
    if probe_type == "linear":
        return LinearProbe(input_dim)
    elif probe_type == "mlp":
        return MLPProbe(input_dim, **kwargs)
    elif probe_type == "conv":
        return ConvProbe(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
