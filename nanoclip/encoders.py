"""The two towers share ordinary pre-norm Transformer blocks, but no weights."""

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, causal: bool = False):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # PyTorch supplies the baseline attention kernel; the projections and block are ours.
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(x.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, causal: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads, causal)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class VisionEncoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        self.image_size = image_size
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position = nn.Parameter(
            torch.randn(1, (image_size // patch_size) ** 2 + 1, width) * 0.02
        )
        self.pre_norm = nn.LayerNorm(width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.post_norm = nn.LayerNorm(width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"expected images [batch, 3, {self.image_size}, {self.image_size}]")
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls.expand(images.shape[0], -1, -1), x), dim=1)
        x = self.blocks(self.pre_norm(x + self.position))
        return self.post_norm(x[:, 0])


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, width: int, layers: int, heads: int):
        super().__init__()
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, context_length, width) * 0.02)
        self.blocks = nn.Sequential(*(Block(width, heads, causal=True) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or not 2 <= tokens.shape[1] <= self.context_length:
            raise ValueError(
                f"expected tokens [batch, length] with 2 <= length <= {self.context_length}"
            )
        # ByteTokenizer always emits exactly one EOS. Causality keeps later PAD tokens
        # out of its representation, so no extra attention padding mask is necessary.
        eos_positions = (tokens == 2).int().argmax(dim=-1)
        x = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        x = self.norm(self.blocks(x))
        return x[torch.arange(tokens.shape[0], device=tokens.device), eos_positions]
