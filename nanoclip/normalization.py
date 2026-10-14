"""The backend boundary for our first GPU experiment; the default is always PyTorch."""

import math

import torch
from torch.nn import functional as F


def l2_normalize(x: torch.Tensor, backend: str = "torch", eps: float = 1e-6) -> torch.Tensor:
    """Normalize rows as x / max(||x||_2, eps), accumulating low precision inputs in fp32.

    Output dtype matches input dtype. The Triton path supports first derivatives only.
    Explicitly requesting Triton never silently falls back to PyTorch.
    """
    if x.ndim != 2 or min(x.shape) == 0 or not x.is_floating_point():
        raise ValueError("normalization expects a nonempty floating point [rows, features] tensor")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if backend == "torch":
        compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return F.normalize(compute, dim=-1, eps=eps).to(x.dtype)
    if backend == "triton":
        if x.device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("the Triton backend requires an NVIDIA CUDA GPU; use --norm torch")
        try:
            from nanoclip.kernels.l2_norm import triton_l2_normalize
        except ImportError as exc:
            raise RuntimeError("Triton is unavailable; install with uv sync --extra gpu") from exc
        return triton_l2_normalize(x, eps)
    raise ValueError(f"unknown normalization backend: {backend}")
