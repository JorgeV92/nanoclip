"""CUDA-event benchmarks, including allocation/wrapper overhead and separate backward timing."""

import argparse
import json

import torch

from nanoclip.normalization import l2_normalize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[256, 4096, 16384])
    parser.add_argument("--dims", nargs="+", type=int, default=[64, 128, 512, 1024])
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--mode", choices=["forward", "backward", "both"], default="both")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("benchmark requires an NVIDIA CUDA GPU")
    try:
        import triton
        import triton.testing
    except ImportError:
        parser.error("install Triton with uv sync --extra gpu")
    if min(args.rows + args.dims) < 1 or max(args.dims) > 16384:
        parser.error("rows/dims must be positive; dims <= 16384")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bf16")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "dtype": args.dtype,
            }
        )
    )
    for rows in args.rows:
        for dim in args.dims:
            x = torch.randn(rows, dim, device="cuda", dtype=dtype, requires_grad=True)
            gradient = torch.randn_like(x)
            # Assert correctness before reporting timings. do_bench warms up compiled kernels.
            reference = l2_normalize(x, "torch")
            custom = l2_normalize(x, "triton")
            atol = {"fp32": 2e-5, "fp16": 2e-3, "bf16": 2e-2}[args.dtype]
            torch.testing.assert_close(custom, reference, atol=atol, rtol=atol)
            for mode in ["forward", "backward"] if args.mode == "both" else [args.mode]:
                timings = {}
                for backend in ("torch", "triton"):
                    if mode == "forward":

                        def operation(backend=backend, x=x):
                            with torch.no_grad():
                                return l2_normalize(x, backend)
                    else:
                        y = l2_normalize(x, backend)

                        def operation(y=y, x=x, gradient=gradient):
                            return torch.autograd.grad(y, x, gradient, retain_graph=True)

                    timings[backend] = triton.testing.do_bench(operation, warmup=100, rep=300)
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "dim": dim,
                            "mode": mode,
                            "torch_ms": timings["torch"],
                            "triton_ms": timings["triton"],
                            "speedup": timings["torch"] / timings["triton"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
