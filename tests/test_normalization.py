import pytest
import torch
from torch.nn import functional as F

from nanoclip.normalization import l2_normalize


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_torch_normalization_outputs_and_gradients(dtype):
    x = torch.randn(5, 17, dtype=dtype).requires_grad_()
    reference = x.detach().clone().requires_grad_()
    compute = reference.float() if dtype in (torch.float16, torch.bfloat16) else reference
    expected = F.normalize(compute, dim=-1, eps=1e-6).to(dtype)
    actual = l2_normalize(x)
    grad = torch.randn_like(x)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference.grad)


def test_clamped_norm_gradcheck_and_zero_rows():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [0.03, 0.04, 0.0], [3.0, 4.0, 0.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(lambda value: l2_normalize(value, eps=0.1), (x,))
    l2_normalize(x, eps=0.1).sum().backward()
    torch.testing.assert_close(x.grad[0], torch.full((3,), 10.0, dtype=torch.double))


def test_backend_errors_are_explicit():
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        l2_normalize(torch.randn(3, 8), backend="triton")
    with pytest.raises(ValueError, match="unknown"):
        l2_normalize(torch.randn(3, 8), backend="unknown")
    for eps in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eps"):
            l2_normalize(torch.randn(3, 8), eps=eps)


def test_noncontiguous_input():
    x = torch.randn(11, 3, requires_grad=True)
    result = l2_normalize(x.T)
    torch.testing.assert_close(result, F.normalize(x.T, dim=-1))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
