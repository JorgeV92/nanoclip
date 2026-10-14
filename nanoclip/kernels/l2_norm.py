"""One Triton program per embedding row, including an explicit vector-Jacobian product.

Let r = ||x||, d = max(r, eps), y = x / d, and g = dL/dy.
For r >= eps: dL/dx = (g - y * sum(g*y)) / d.
For r <  eps: dL/dx = g / eps (the denominator is constant).
At equality we follow PyTorch clamp_min's derivative of 1. Save x rather than rounded
fp16/bf16 y, so backward reconstructs the normalized vector in fp32.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _forward(X, Y, INV, ACTIVE, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    inv = 1.0 / tl.maximum(norm, EPS)
    tl.store(Y + row * D + col, x * inv, col < D)
    tl.store(INV + row, inv)
    tl.store(ACTIVE + row, norm >= EPS)


@triton.jit
def _backward(X, G, DX, INV, ACTIVE, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    g = tl.load(G + row * D + col, col < D, other=0).to(tl.float32)
    inv = tl.load(INV + row)
    active = tl.load(ACTIVE + row)
    y = x * inv
    dot = tl.sum(g * y, axis=0)
    dx = (g - tl.where(active, y * dot, 0.0)) * inv
    tl.store(DX + row * D + col, dx, col < D)


class _L2Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        rows, dim = x.shape
        y = torch.empty_like(x)
        inv = torch.empty(rows, device=x.device, dtype=torch.float32)
        active = torch.empty(rows, device=x.device, dtype=torch.bool)
        block = triton.next_power_of_2(dim)
        warps = 4 if block <= 1024 else 8
        # Select x's device even when the process's current CUDA device differs.
        with torch.cuda.device(x.device):
            _forward[(rows,)](x, y, inv, active, dim, eps, block, num_warps=warps)
        ctx.save_for_backward(x, inv, active)
        ctx.block, ctx.warps = block, warps
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, inv, active = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dx = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _backward[(x.shape[0],)](
                x, grad_output, dx, inv, active, x.shape[1], ctx.block, num_warps=ctx.warps
            )
        return dx, None


def triton_l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Triton L2 normalization supports float32, float16, and bfloat16")
    if x.shape[1] > 16384:
        raise ValueError("this educational row kernel supports at most 16384 features")
    # A copy, when required, remains in the autograd graph outside the custom Function.
    return _L2Normalize.apply(x.contiguous(), eps)
