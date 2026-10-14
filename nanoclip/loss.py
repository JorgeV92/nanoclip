"""Symmetric in-batch image/text contrastive cross entropy."""

import math

import torch
from torch.nn import functional as F


def clip_loss(images: torch.Tensor, texts: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """The diagonal contains positives; all other pairs in this batch are negatives."""
    if images.ndim != 2 or images.shape != texts.shape or images.shape[0] < 2:
        raise ValueError("CLIP loss needs matching [batch, embedding] tensors and batch >= 2")
    # Disabling autocast here makes both the similarity matrix and CE genuinely fp32.
    with torch.autocast(device_type=images.device.type, enabled=False):
        scale = logit_scale.float().clamp(max=math.log(100)).exp()
        logits = scale * (images.float() @ texts.float().T)
        targets = torch.arange(images.shape[0], device=images.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) * 0.5
