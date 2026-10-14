"""A few explicit device / precision choices, shared by the command-line scripts."""

from contextlib import nullcontext

import torch


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if name == "cuda" and (not torch.cuda.is_available() or torch.version.hip is not None):
        raise ValueError("CUDA requested but no NVIDIA GPU is available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but it is not available")
    return torch.device(name)


def select_precision(name: str, device: torch.device) -> str:
    if name == "auto":
        return (
            ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            if device.type == "cuda"
            else "fp32"
        )
    if name != "fp32" and device.type != "cuda":
        raise ValueError(
            "this first iteration supports mixed precision on CUDA; use fp32 on CPU/MPS"
        )
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 is unsupported on this GPU; use fp16 or fp32")
    return name


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )
