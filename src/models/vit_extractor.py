"""HuggingFace ViT wrapper for hidden state extraction."""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTImageProcessor
from typing import Optional


class ViTExtractor:
    """Wrapper around HuggingFace ViT for extracting hidden states at all layers.

    Supports both pretrained and randomly initialized models.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cpu")

        if pretrained:
            self.model = ViTModel.from_pretrained(
                model_name, output_hidden_states=True
            )
        else:
            # Same architecture, random weights
            config = ViTConfig.from_pretrained(model_name)
            self.model = ViTModel(config)
            self.model.config.output_hidden_states = True

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image processor for proper preprocessing
        self.processor = ViTImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from all layers.

        Args:
            pixel_values: Preprocessed image tensor (B, 3, 224, 224).

        Returns:
            Tensor of shape (num_layers, B, 196, 768) containing patch token
            hidden states from each layer. Layer 0 = patch embedding output,
            layers 1-12 = transformer block outputs.
            CLS token is excluded (only patch tokens kept).
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=pixel_values)

        # hidden_states is a tuple of (num_layers+1,) tensors, each (B, 197, 768)
        # Index 0 = embedding layer output, 1-12 = transformer block outputs
        hidden_states = outputs.hidden_states  # tuple of 13 tensors

        # Stack and remove CLS token (index 0), keep patch tokens (indices 1:197)
        stacked = torch.stack(hidden_states, dim=0)  # (13, B, 197, 768)
        patch_tokens = stacked[:, :, 1:, :]  # (13, B, 196, 768)

        return patch_tokens

    @torch.no_grad()
    def extract_single(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract hidden states for a single image.

        Args:
            pixel_values: Single image tensor (3, 224, 224) or (1, 3, 224, 224).

        Returns:
            Tensor of shape (13, 196, 768).
        """
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        result = self.extract(pixel_values)  # (13, 1, 196, 768)
        return result.squeeze(1)  # (13, 196, 768)
