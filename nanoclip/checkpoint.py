"""Portable, atomic checkpoints: model, optimizer, scaler, cursor, and RNG states."""

import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

from nanoclip.model import CLIP, CLIPConfig


def capture_rng() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        # This project uses one GPU. A different instance may expose a different GPU count.
        for index, value in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value, index)
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path, model, optimizer, scaler, step, epoch, batch_in_epoch, train_args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "rng": capture_rng(),
        "train_args": train_args,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_checkpoint(path) -> dict:
    # This format consists only of tensors and plain containers, so no arbitrary unpickling.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return payload


def load_model(path, device="cpu", norm_backend="torch") -> CLIP:
    payload = read_checkpoint(path)
    model = CLIP(CLIPConfig(**payload["config"]), norm_backend)
    model.load_state_dict(payload["model"])
    return model.to(device).eval()
