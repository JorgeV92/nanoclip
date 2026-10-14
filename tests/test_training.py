import json

import pytest
import torch

from nanoclip.checkpoint import load_model, read_checkpoint
from nanoclip.dataset import SyntheticDataset
from nanoclip.loss import clip_loss
from nanoclip.model import CLIP, CLIPConfig
from nanoclip.tokenizer import ByteTokenizer
from scripts.train import parse_args, run_training


def train(tmp_path, name, steps, extra=()):
    return run_training(
        parse_args(
            [
                "--device",
                "cpu",
                "--threads",
                "1",
                "--samples",
                "22",
                "--val-samples",
                "7",
                "--batch-size",
                "4",
                "--grad-accum",
                "3",
                "--steps",
                str(steps),
                "--log-every",
                "1",
                "--save-every",
                "1",
                "--eval-every",
                "1",
                "--output-dir",
                str(tmp_path / name),
                *extra,
            ]
        )
    )


def test_training_checkpoint_and_exact_resume_across_epoch_boundary(
    tmp_path, mini_config, monkeypatch
):
    monkeypatch.setattr(CLIPConfig, "from_size", classmethod(lambda cls, size: mini_config))
    full = train(tmp_path, "full", 4)
    partial = train(tmp_path, "split", 1)
    partial_state = read_checkpoint(partial)
    assert partial_state["batch_in_epoch"] == 3
    # Resume defaults come from the checkpoint, including batch size and accumulation.
    resumed = run_training(
        parse_args(["--device", "cpu", "--threads", "1", "--resume", str(partial), "--steps", "4"])
    )
    a, b = read_checkpoint(full), read_checkpoint(resumed)
    assert a["step"] == b["step"] == 4
    assert a["epoch"] == b["epoch"] == 2
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    for parameter in a["optimizer"]["state"]:
        for key in a["optimizer"]["state"][parameter]:
            torch.testing.assert_close(
                a["optimizer"]["state"][parameter][key],
                b["optimizer"]["state"][parameter][key],
                atol=0,
                rtol=0,
            )
    records = [
        json.loads(line) for line in (full.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["microbatches"] for row in records if "train_loss" in row] == [3, 2, 3, 2]
    assert all(row["val_pairs"] == 7 for row in records if "val_pairs" in row)
    assert load_model(full).config == mini_config


def test_resume_rejects_changed_batch_size(tmp_path, mini_config, monkeypatch):
    monkeypatch.setattr(CLIPConfig, "from_size", classmethod(lambda cls, size: mini_config))
    checkpoint = train(tmp_path, "run", 1)
    with pytest.raises(ValueError, match="batch_size"):
        run_training(parse_args(["--resume", str(checkpoint), "--steps", "2", "--batch-size", "8"]))


def test_training_rejects_no_full_batches(tmp_path):
    args = parse_args(
        ["--device", "cpu", "--samples", "2", "--batch-size", "4", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="at least --batch-size"):
        run_training(args)


def test_accumulation_matches_mean_objective_including_short_window(
    tmp_path, mini_config, monkeypatch
):
    monkeypatch.setattr(CLIPConfig, "from_size", classmethod(lambda cls, size: mini_config))
    snapshots = []
    clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def capture_gradients(parameters, max_norm):
        parameters = list(parameters)
        snapshots.append([(p.detach().clone(), p.grad.detach().clone()) for p in parameters])
        return clip_grad_norm(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_gradients)
    train(tmp_path, "accumulation", 2)
    dataset = SyntheticDataset(22, 16, ByteTokenizer(mini_config.context_length), seed=42)
    order = torch.randperm(22, generator=torch.Generator().manual_seed(42)).tolist()[:20]
    # Independent reference: construct a single averaged objective before calling backward.
    # Compare gradients at the same parameters, before AdamW amplifies tiny roundoff
    # in theoretically zero gradients (e.g. attention's key bias).
    for window, snapshot in zip((order[:12], order[12:]), snapshots, strict=True):
        model = CLIP(mini_config)
        with torch.no_grad():
            for parameter, (value, _) in zip(model.parameters(), snapshot, strict=True):
                parameter.copy_(value)
        losses = []
        for start in range(0, len(window), 4):
            pairs = [dataset[i] for i in window[start : start + 4]]
            pixels = torch.stack([pair[0] for pair in pairs])
            tokens = torch.stack([pair[1] for pair in pairs])
            losses.append(clip_loss(*model(pixels, tokens)))
        torch.stack(losses).mean().backward()
        for parameter, (_, actual) in zip(model.parameters(), snapshot, strict=True):
            torch.testing.assert_close(actual, parameter.grad, atol=2e-7, rtol=2e-4)
