"""Evaluate contrastive loss and image/text retrieval over a held-out dataset."""

import argparse
import json

from torch.utils.data import DataLoader

from nanoclip.checkpoint import load_model
from nanoclip.dataset import make_dataset
from nanoclip.evaluation import evaluate
from nanoclip.runtime import select_device, select_precision
from nanoclip.tokenizer import ByteTokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", help="held-out JSONL; omit for synthetic evaluation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1_000_042)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--norm", choices=["torch", "triton"], default="torch")
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    device = select_device(args.device)
    model = load_model(args.checkpoint, device, args.norm)
    dataset = make_dataset(
        args.data,
        model.config.image_size,
        ByteTokenizer(model.config.context_length),
        args.samples,
        args.seed,
    )
    metrics = evaluate(
        model,
        DataLoader(dataset, batch_size=args.batch_size),
        device,
        select_precision(args.precision, device),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
