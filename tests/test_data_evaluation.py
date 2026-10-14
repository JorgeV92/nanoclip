import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from nanoclip.dataset import CaptionDataset, SyntheticDataset
from nanoclip.evaluation import retrieval_metrics
from nanoclip.tokenizer import ByteTokenizer
from scripts.prepare_data import main as prepare_data


def test_synthetic_data_is_deterministic_and_paired():
    tokenizer = ByteTokenizer(64)
    first = SyntheticDataset(4, 32, tokenizer, seed=10)
    second = SyntheticDataset(4, 32, tokenizer, seed=10)
    for index in range(4):
        pixels, tokens, group = first[index]
        torch.testing.assert_close(pixels, second[index][0])
        torch.testing.assert_close(tokens, second[index][1])
        assert pixels.shape == (3, 32, 32)
        assert -1 <= pixels.min() <= pixels.max() <= 1
        assert tokenizer.decode(tokens.tolist()).startswith("a ")
        assert group == index


def test_jsonl_dataset_relative_paths_grayscale_and_repeated_images(tmp_path):
    Image.new("L", (20, 10), 255).save(tmp_path / "image.png")
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"image": "image.png", "caption": caption})
            for caption in ["a white image", "a bright image"]
        )
    )
    dataset = CaptionDataset(manifest, 16, ByteTokenizer())
    assert dataset[0][0].shape == (3, 16, 16)
    assert dataset[0][2] == dataset[1][2]
    assert torch.equal(dataset[0][0], torch.ones(3, 16, 16))


def test_bad_manifest_reports_line(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text('{"image": "missing.png", "caption": "test"}\n')
    with pytest.raises(ValueError, match="pairs.jsonl:1.*missing image"):
        CaptionDataset(manifest, 16, ByteTokenizer())


def test_prepare_local_dataset(tmp_path):
    prepare_data(["--output", str(tmp_path), "--samples", "16", "--image-size", "32"])
    train = CaptionDataset(tmp_path / "train.jsonl", 16, ByteTokenizer())
    val = CaptionDataset(tmp_path / "val.jsonl", 16, ByteTokenizer())
    assert len(train) + len(val) == 16
    assert {row[0] for row in train.records}.isdisjoint(row[0] for row in val.records)


def test_flickr_importer_uses_one_caption_per_image_and_handles_short_stream(tmp_path, monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (20, 10), (i * 30, 0, 0)),
            "caption_0": f"image {i}",
            "caption_1": "unused caption",
        }
        for i in range(6)
    ]

    def load_dataset(name, *, split, streaming):
        assert (name, split, streaming) == ("jxie/flickr8k", "train", True)
        return SimpleNamespace(take=lambda count: iter(rows[:count]))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    prepare_data(["--source", "flickr8k", "--samples", "100", "--output", str(tmp_path)])
    train = CaptionDataset(tmp_path / "train.jsonl", 16, ByteTokenizer())
    val = CaptionDataset(tmp_path / "val.jsonl", 16, ByteTokenizer())
    assert len(train) == 4 and len(val) == 2
    assert {row[1] for row in train.records + val.records} == {f"image {i}" for i in range(6)}


def test_retrieval_uses_global_gallery_and_supports_multiple_captions():
    images = torch.eye(6)
    result = retrieval_metrics(images, images, chunk_size=2)
    assert all(value == 1.0 for value in result.values())
    shuffled = retrieval_metrics(images, images.roll(1, 0), chunk_size=2)
    assert shuffled["image_to_text_r@1"] == 0
    assert shuffled["text_to_image_r@1"] == 0
    texts = images.repeat_interleave(2, dim=0)
    ids = torch.arange(6)
    result = retrieval_metrics(images, texts, ids, ids.repeat_interleave(2), chunk_size=3)
    assert all(value == 1.0 for value in result.values())
