"""CLIP in two small configurations. All model weights are randomly initialized."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from nanoclip.encoders import TextEncoder, VisionEncoder
from nanoclip.normalization import l2_normalize


@dataclass(frozen=True)
class CLIPConfig:
    image_size: int = 64
    patch_size: int = 8
    width: int = 128
    vision_layers: int = 2
    text_layers: int = 2
    heads: int = 4
    embed_dim: int = 64
    context_length: int = 64
    vocab_size: int = 259

    def __post_init__(self):
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("model dimensions must be positive")
        if self.image_size % self.patch_size or self.width % self.heads:
            raise ValueError("image_size must divide into patches and width must divide into heads")
        if self.context_length < 2 or self.vocab_size != 259:
            raise ValueError(
                "the byte tokenizer requires context_length >= 2 and vocab_size == 259"
            )

    @classmethod
    def from_size(cls, size: str) -> "CLIPConfig":
        if size == "tiny":
            return cls()
        if size == "small":
            return cls(
                image_size=128,
                patch_size=16,
                width=256,
                vision_layers=4,
                text_layers=4,
                heads=8,
                embed_dim=128,
                context_length=96,
            )
        raise ValueError(f"unknown model size: {size}")


class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig | None = None, norm_backend: str = "torch"):
        super().__init__()
        self.config = config or CLIPConfig()
        self.norm_backend = norm_backend
        c = self.config
        self.vision = VisionEncoder(c.image_size, c.patch_size, c.width, c.vision_layers, c.heads)
        self.text = TextEncoder(c.vocab_size, c.context_length, c.width, c.text_layers, c.heads)
        self.image_projection = nn.Linear(c.width, c.embed_dim, bias=False)
        self.text_projection = nn.Linear(c.width, c.embed_dim, bias=False)
        # Store log(inverse temperature), initialized to the CLIP paper's 0.07 temperature.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.image_projection(self.vision(images)), self.norm_backend)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.text_projection(self.text(tokens)), self.norm_backend)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        return self.encode_image(images), self.encode_text(tokens), self.logit_scale
