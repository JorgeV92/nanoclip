"""Deterministic image preprocessing and paired synthetic / local JSONL datasets."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

from nanoclip.tokenizer import ByteTokenizer


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """Center-crop to a square, resize, map RGB [0, 255] to [-1, 1]. No pretrained stats."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
    array = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(127.5).sub_(1)


def synthetic_pair(index: int, image_size: int, seed: int = 0) -> tuple[Image.Image, str]:
    """Render captioned shapes, so smoke-test pairs actually contain learnable signal."""
    rng = random.Random(seed + index)
    colors = {
        "red": (230, 45, 45),
        "green": (40, 200, 70),
        "blue": (40, 90, 230),
        "yellow": (240, 210, 40),
        "purple": (170, 50, 220),
        "orange": (240, 140, 30),
    }
    color = rng.choice(list(colors))
    shape = rng.choice(["circle", "square", "triangle"])
    horizontal = rng.choice(["left", "center", "right"])
    vertical = rng.choice(["top", "middle", "bottom"])
    size = rng.choice(["small", "large"])
    radius = image_size * (0.12 if size == "small" else 0.21)
    cx = image_size * {"left": 0.25, "center": 0.5, "right": 0.75}[horizontal]
    cy = image_size * {"top": 0.25, "middle": 0.5, "bottom": 0.75}[vertical]
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    image = Image.new("RGB", (image_size, image_size), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse(box, fill=colors[color])
    elif shape == "square":
        draw.rectangle(box, fill=colors[color])
    else:
        draw.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
            fill=colors[color],
        )
    return image, f"a {size} {color} {shape} at {vertical} {horizontal}"


class SyntheticDataset(Dataset):
    def __init__(self, samples: int, image_size: int, tokenizer: ByteTokenizer, seed: int = 0):
        if samples < 2:
            raise ValueError("a contrastive dataset needs at least two pairs")
        self.samples, self.image_size, self.tokenizer, self.seed = (
            samples,
            image_size,
            tokenizer,
            seed,
        )

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        image, caption = synthetic_pair(index, self.image_size, self.seed)
        return (
            preprocess_image(image, self.image_size),
            torch.tensor(self.tokenizer.encode(caption), dtype=torch.long),
            index,
        )


class CaptionDataset(Dataset):
    """JSONL: {"image": "images/001.jpg", "caption": "a cat", "image_id": "001"}.

    Paths are relative to the manifest, unless absolute. image_id is optional; by default
    the resolved image path groups multiple captions for retrieval evaluation.
    """

    def __init__(self, manifest: str | Path, image_size: int, tokenizer: ByteTokenizer):
        self.manifest = Path(manifest).resolve()
        self.image_size, self.tokenizer = image_size, tokenizer
        self.records = []
        group_ids = {}
        with self.manifest.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    image, caption = record["image"], record["caption"]
                    if not isinstance(image, str) or not isinstance(caption, str) or not caption:
                        raise ValueError("image and nonempty caption must be strings")
                    path = (self.manifest.parent / image).resolve()
                    if not path.is_file():
                        raise ValueError(f"missing image: {path}")
                    key = record.get("image_id", str(path))
                    if not isinstance(key, (str, int)):
                        raise ValueError("image_id must be a string or integer")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{self.manifest}:{line_number}: {exc}") from exc
                group_id = group_ids.setdefault(key, len(group_ids))
                self.records.append((path, caption, group_id))
        if len(self.records) < 2:
            raise ValueError("a caption manifest needs at least two records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, caption, group_id = self.records[index]
        with Image.open(path) as image:
            pixels = preprocess_image(image, self.image_size)
        return pixels, torch.tensor(self.tokenizer.encode(caption), dtype=torch.long), group_id


def make_dataset(manifest, image_size, tokenizer, samples=256, seed=0):
    if manifest is not None:
        return CaptionDataset(manifest, image_size, tokenizer)
    return SyntheticDataset(samples, image_size, tokenizer, seed)
