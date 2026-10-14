"""Create local JSONL pairs, either offline shapes or a small public caption dataset."""

import argparse
import json
import random
from pathlib import Path

from nanoclip.dataset import synthetic_pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["shapes", "flickr8k"], default="shapes")
    parser.add_argument("--output", default="data/shapes")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.samples < 4 or not 0 < args.val_fraction < 1 or args.image_size < 8:
        parser.error("need samples >= 4, image-size >= 8, and 0 < val-fraction < 1")
    output = Path(args.output)
    if (output / "train.jsonl").exists() or (output / "val.jsonl").exists():
        parser.error("output already has manifests; choose a new directory")
    if args.source == "flickr8k":
        try:
            from datasets import load_dataset
        except ImportError:
            parser.error("install the optional data dependencies: uv sync --extra cpu --extra data")
        dataset = load_dataset("jxie/flickr8k", split="train", streaming=True)
        # One caption per image avoids false negatives from repeated images in training.
        pairs = ((row["image"], row["caption_0"]) for row in dataset.take(args.samples))
    else:
        pairs = (synthetic_pair(i, args.image_size, args.seed) for i in range(args.samples))
    (output / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (image, caption) in enumerate(pairs):
        relative = f"images/{index:06d}.png"
        # Save original aspect ratio for the same center-crop preprocessing at every run.
        image.convert("RGB").save(output / relative)
        rows.append({"image": relative, "caption": caption})
    if len(rows) < 4:
        parser.error("dataset needs at least four examples for a train/validation split")
    val_count = max(2, min(len(rows) - 2, round(len(rows) * args.val_fraction)))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    val_indices = set(order[:val_count])
    records = {"train": [], "val": []}
    for index, row in enumerate(rows):
        split = "val" if index in val_indices else "train"
        records[split].append(row)
    for split, rows in records.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    print(
        f"wrote {len(records['train'])} train and {len(records['val'])} "
        f"validation pairs to {output}"
    )


if __name__ == "__main__":
    main()
