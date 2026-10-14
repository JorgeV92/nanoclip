"""Single-device CLIP training. Read top to bottom: arguments, data, model, loop."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanoclip.checkpoint import read_checkpoint, restore_rng, save_checkpoint
from nanoclip.dataset import make_dataset
from nanoclip.evaluation import evaluate
from nanoclip.loss import clip_loss
from nanoclip.model import CLIP, CLIPConfig
from nanoclip.runtime import autocast_context, select_device, select_precision
from nanoclip.tokenizer import ByteTokenizer

DEFAULTS = dict(
    size="tiny",
    data=None,
    val_data=None,
    samples=256,
    val_samples=64,
    batch_size=16,
    grad_accum=1,
    steps=1000,
    lr=3e-4,
    weight_decay=0.01,
    seed=42,
    precision="auto",
    norm="torch",
    workers=0,
    log_every=10,
    eval_every=100,
    save_every=100,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["tiny", "small"])
    parser.add_argument("--data", help="training JSONL; omit for captioned synthetic shapes")
    parser.add_argument("--val-data", help="held-out JSONL, required when --data is given")
    for name in (
        "samples",
        "val_samples",
        "batch_size",
        "grad_accum",
        "steps",
        "seed",
        "workers",
        "log_every",
        "eval_every",
        "save_every",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--norm", choices=["torch", "triton"])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count")
    parser.add_argument("--output-dir", help="defaults to checkpoint's directory on resume")
    parser.add_argument("--resume", help="checkpoint; --steps is the total target step count")
    return parser.parse_args(argv)


def resolve_args(args, checkpoint):
    saved = checkpoint["train_args"] if checkpoint else {}
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, value))
    if args.output_dir is None:
        args.output_dir = str(Path(args.resume).parent) if args.resume else "checkpoints/demo"
    for key in ("batch_size", "samples", "val_samples"):
        if getattr(args, key) < 2:
            raise ValueError(f"{key} must be >= 2 for contrastive training")
    for key in ("grad_accum", "steps", "log_every", "eval_every", "save_every", "threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("workers/weight_decay must be nonnegative and lr must be positive")
    if not math.isfinite(args.lr) or not math.isfinite(args.weight_decay):
        raise ValueError("optimizer settings must be finite")
    if bool(args.data) != bool(args.val_data):
        raise ValueError("provide both --data and --val-data, or neither for synthetic data")
    if args.data and Path(args.data).resolve() == Path(args.val_data).resolve():
        raise ValueError("use separate training and held-out validation manifests")
    if checkpoint:
        for key in ("size", "batch_size", "grad_accum", "seed", "samples", "lr", "weight_decay"):
            if getattr(args, key) != saved[key]:
                raise ValueError(f"cannot change {key} on resume; start a fresh run instead")
        if args.steps <= checkpoint["step"]:
            raise ValueError("--steps must exceed the saved step (it is a total, not extra steps)")


def dataset_signature(args):
    if args.data:
        return hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    return f"synthetic:{args.seed}:{args.samples}"


def run_training(args):
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    resolve_args(args, checkpoint)
    device = select_device(args.device)
    precision = select_precision(args.precision, device)
    if args.norm == "triton" and device.type != "cuda":
        raise ValueError("--norm triton requires --device cuda")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = CLIPConfig(**checkpoint["config"]) if checkpoint else CLIPConfig.from_size(args.size)
    tokenizer = ByteTokenizer(config.context_length)
    train_data = make_dataset(args.data, config.image_size, tokenizer, args.samples, args.seed)
    val_data = make_dataset(
        args.val_data, config.image_size, tokenizer, args.val_samples, args.seed + 1_000_000
    )
    batches_per_epoch = len(train_data) // args.batch_size
    if batches_per_epoch == 0:
        raise ValueError("training dataset must contain at least --batch-size examples")
    signature = dataset_signature(args)
    if checkpoint and checkpoint["train_args"]["dataset_signature"] != signature:
        raise ValueError(
            "training data changed since the checkpoint (manifest contents must match)"
        )
    # A separate generator prevents DataLoader worker seeding from consuming model RNG.
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    model = CLIP(config, args.norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    step, epoch, batch_in_epoch = 0, 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        step, epoch, batch_in_epoch = (
            checkpoint[key] for key in ("step", "epoch", "batch_in_epoch")
        )
        restore_rng(checkpoint["rng"])
    output = Path(args.output_dir)
    if (output / "latest.pt").exists() and checkpoint is None:
        raise ValueError(f"{output}/latest.pt exists; use --resume or a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    train_args = {**vars(args), "dataset_signature": signature}
    (output / "config.json").write_text(json.dumps(train_args, indent=2) + "\n")
    print(
        json.dumps(
            {
                "device": str(device),
                "precision": precision,
                "norm": args.norm,
                "parameters": sum(p.numel() for p in model.parameters()),
                "start_step": step,
                "target_step": args.steps,
            }
        ),
        flush=True,
    )

    def log(record):
        print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    while step < args.steps:
        # Reconstruct the permutation from (seed, epoch), then start at the saved cursor.
        # Checkpoints are taken only after optimizer steps, never half way through accumulation.
        order = torch.randperm(
            len(train_data), generator=torch.Generator().manual_seed(args.seed + epoch)
        ).tolist()
        order = order[batch_in_epoch * args.batch_size : batches_per_epoch * args.batch_size]
        loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=order,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        iterator = iter(loader)
        remaining = batches_per_epoch - batch_in_epoch
        while remaining and step < args.steps:
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            microbatches = min(args.grad_accum, remaining)
            total_loss = 0.0
            for _ in range(microbatches):
                pixels, tokens, _ = next(iterator)
                with autocast_context(device, precision):
                    loss = clip_loss(*model(pixels.to(device), tokens.to(device)))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step + 1}")
                # Average the actual number of microbatches, including a short final window.
                scaler.scale(loss / microbatches).backward()
                total_loss += loss.item()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if precision != "fp16" and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite gradient at step {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(min=math.log(1e-3), max=math.log(100))
            step += 1
            batch_in_epoch += microbatches
            remaining -= microbatches
            if remaining == 0:
                epoch, batch_in_epoch = epoch + 1, 0
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                log(
                    {
                        "step": step,
                        "train_loss": total_loss / microbatches,
                        "logit_scale": model.logit_scale.exp().item(),
                        "seconds": time.perf_counter() - start,
                        "microbatches": microbatches,
                    }
                )
            if step % args.eval_every == 0 or step == args.steps:
                log({"step": step, **evaluate(model, val_loader, device, precision)})
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(
                    output / "latest.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    batch_in_epoch,
                    train_args,
                )
    return output / "latest.pt"


def main(argv=None):
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
