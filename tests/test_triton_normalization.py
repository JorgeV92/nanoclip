"""These are real CUDA tests: no interpreter mode and no CPU emulation of the kernel."""

import importlib.util

import pytest
import torch

from nanoclip.loss import clip_loss
from nanoclip.model import CLIP
from nanoclip.normalization import l2_normalize
from nanoclip.tokenizer import ByteTokenizer

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or importlib.util.find_spec("triton") is None,
        reason="requires an NVIDIA CUDA GPU and Triton",
    ),
]


@pytest.mark.parametrize("shape", [(1, 1), (7, 3), (17, 64), (33, 127), (8, 256), (3, 1537)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_matches_pytorch_forward_and_backward(shape, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bf16")
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    # A noncontiguous upstream gradient also exercises the wrapper's layout handling.
    grad = torch.randn(shape[::-1], device="cuda", dtype=dtype).T
    actual = l2_normalize(x, "triton")
    expected = l2_normalize(reference, "torch")
    actual.backward(grad)
    expected.backward(grad)
    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x.grad, reference.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("eps", [1e-6, 0.25])
def test_zero_below_equal_and_above_epsilon(eps):
    x = torch.zeros(4, 7, device="cuda")
    x[:, 0] = torch.tensor([0.0, eps / 2, eps, eps * 2], device="cuda")
    x.requires_grad_()
    reference = x.detach().clone().requires_grad_()
    grad = torch.randn_like(x) * eps
    actual, expected = l2_normalize(x, "triton", eps), l2_normalize(reference, "torch", eps)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(x.grad, reference.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x.grad[0], grad[0] / eps)


def test_noncontiguous_input_gradient_reaches_base_tensor():
    x = torch.randn(13, 7, device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    grad = torch.randn(7, 13, device="cuda")
    l2_normalize(x.T, "triton").backward(grad)
    l2_normalize(reference.T, "torch").backward(grad)
    torch.testing.assert_close(x.grad, reference.grad, atol=2e-5, rtol=2e-5)


def test_entire_model_loss_and_parameter_gradients(mini_config):
    model = CLIP(mini_config, "triton").cuda()
    reference = CLIP(mini_config, "torch").cuda()
    reference.load_state_dict(model.state_dict())
    pixels = torch.randn(4, 3, 16, 16, device="cuda")
    tokens = ByteTokenizer(mini_config.context_length)(["red", "blue", "green", "yellow"]).cuda()
    actual, expected = clip_loss(*model(pixels, tokens)), clip_loss(*reference(pixels, tokens))
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=5e-4, rtol=5e-4)
