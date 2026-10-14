import pytest
import torch

from nanoclip.model import CLIPConfig


@pytest.fixture(autouse=True)
def deterministic_tests():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def mini_config():
    return CLIPConfig(
        image_size=16,
        patch_size=4,
        width=32,
        heads=4,
        vision_layers=1,
        text_layers=1,
        embed_dim=16,
        context_length=48,
    )
