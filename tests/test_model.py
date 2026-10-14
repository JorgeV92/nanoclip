import math

import pytest
import torch
from torch.nn import functional as F

from nanoclip.loss import clip_loss
from nanoclip.model import CLIP, CLIPConfig
from nanoclip.tokenizer import ByteTokenizer


def test_tokenizer_utf8_truncation_and_padding():
    tokenizer = ByteTokenizer(16)
    assert tokenizer.decode(tokenizer.encode("café 🐈")) == "café 🐈"
    for text in ("", "a", "x" * 100):
        tokens = tokenizer.encode(text)
        assert len(tokens) == 16
        assert tokens[0] == tokenizer.bos_id
        assert tokens.count(tokenizer.eos_id) == 1
    assert tokenizer.encode("x" * 100)[-1] == tokenizer.eos_id
    assert tokenizer([]).shape == (0, 16)
    with pytest.raises(ValueError):
        ByteTokenizer(1)


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_public_model_sizes(size):
    config = CLIPConfig.from_size(size)
    model = CLIP(config)
    with torch.no_grad():
        images, texts, scale = model(
            torch.randn(2, 3, config.image_size, config.image_size),
            ByteTokenizer(config.context_length)(["red cat", "blue dog"]),
        )
    assert images.shape == texts.shape == (2, config.embed_dim)
    torch.testing.assert_close(images.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(texts.norm(dim=-1), torch.ones(2))
    assert scale.exp().item() == pytest.approx(1 / 0.07)


def test_gradients_reach_both_encoders_and_temperature(mini_config):
    model = CLIP(mini_config)
    pixels = torch.randn(4, 3, 16, 16)
    tokens = ByteTokenizer(mini_config.context_length)(["red", "green", "blue", "yellow"])
    clip_loss(*model(pixels, tokens)).backward()
    for parameter in (
        model.vision.patch_embed.weight,
        model.text.embedding.weight,
        model.image_projection.weight,
        model.text_projection.weight,
        model.logit_scale,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_text_pooling_cannot_see_padding_or_later_tokens(mini_config):
    model = CLIP(mini_config).eval()
    tokens = ByteTokenizer(mini_config.context_length)(["cat", "dog"])
    altered = tokens.clone()
    altered[:, 5:] = torch.randint(3, 259, altered[:, 5:].shape)
    torch.testing.assert_close(model.encode_text(tokens), model.encode_text(altered))
    torch.testing.assert_close(model.encode_text(tokens[:, :5]), model.encode_text(tokens))


def test_clip_loss_matches_manual_symmetric_cross_entropy():
    images = F.normalize(torch.randn(5, 8), dim=-1)
    texts = F.normalize(torch.randn(5, 8), dim=-1)
    scale = torch.tensor(math.log(3.0), requires_grad=True)
    logits = 3 * images @ texts.T
    expected = -(logits.log_softmax(0).diag().mean() + logits.log_softmax(1).diag().mean()) / 2
    torch.testing.assert_close(clip_loss(images, texts, scale), expected)
    torch.testing.assert_close(clip_loss(images, texts, scale), clip_loss(texts, images, scale))
    assert clip_loss(torch.eye(4), torch.eye(4), scale) < clip_loss(
        torch.eye(4), torch.eye(4).roll(1, 0), scale
    )
    assert torch.isfinite(clip_loss(images, texts, torch.tensor(1000.0)))


def test_loss_rejects_no_negatives():
    with pytest.raises(ValueError, match="batch >= 2"):
        clip_loss(torch.ones(1, 8), torch.ones(1, 8), torch.tensor(0.0))


def test_overfit_a_fixed_synthetic_batch(mini_config):
    from nanoclip.dataset import SyntheticDataset

    dataset = SyntheticDataset(8, 16, ByteTokenizer(mini_config.context_length), seed=51)
    pixels = torch.stack([dataset[i][0] for i in range(8)])
    tokens = torch.stack([dataset[i][1] for i in range(8)])
    model = CLIP(mini_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = clip_loss(*model(pixels, tokens)).item()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = clip_loss(*model(pixels, tokens))
        loss.backward()
        optimizer.step()
    final = clip_loss(*model(pixels, tokens)).item()
    assert final < initial * 0.4, (initial, final)
