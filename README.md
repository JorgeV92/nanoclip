# nanoclip

A small CLIP implementation for learning training and GPU programming. Inspired by
[nanochat's readable, end-to-end experiments](https://github.com/karpathy/nanochat).
The ViT, text Transformer, projections, byte tokenizer, contrastive objective, and
training loop are implemented here. All weights start randomly; no pretrained CLIP
weights, external CLIP implementations, or training frameworks are used.

The default path is PyTorch on CPU, Apple Silicon (MPS), or CUDA. The first GPU
experiment is a Triton L2-normalization kernel with its own backward pass. Triton
is imported only when requested. This is an educational starting point: small
datasets and short runs will not produce a general-purpose image understanding model.

## Architecture

```text
RGB image -> patch convolution -> CLS + positions -> ViT ---------> projection --\
                                                                              L2 -> similarities -> loss
UTF-8 text -> byte tokens + positions -> causal Transformer -> EOS -> projection --/
```

Both towers use pre-norm residual attention/MLP blocks, learned positions, GELU,
and separate parameters. Attention uses PyTorch's scaled dot-product attention;
the Q/K/V projections and Transformer blocks are explicit in `encoders.py`.
Images are center-cropped, resized, and mapped to [-1, 1]. The byte vocabulary is
259 entries: PAD, BOS, EOS, and 256 byte values. Context limits include BOS/EOS;
long captions are truncated by bytes, with EOS always retained. The causal text
mask keeps trailing padding out of the EOS representation.

| `--size` | Image / patch | Tower width | Vision / text layers | Heads | Embedding | Text context |
|---|---|---|---|---|---|---|
| `tiny` | 64 / 8 | 128 | 2 / 2 | 4 | 64 | 64 bytes/tokens |
| `small` | 128 / 16 | 256 | 4 / 4 | 8 | 128 | 96 bytes/tokens |

`tiny` has 884,609 parameters; `small` has 6,689,537.

For normalized image embeddings `I` and text embeddings `T`, the logits are
`exp(logit_scale) * I @ T.T`. The diagonal contains positive pairs. The loss is
the mean of image-to-text and text-to-image cross entropy. The learnable inverse
temperature starts at `1 / 0.07` and is capped at 100. Similarities and cross
entropy use fp32 even during mixed-precision training.

## Install with uv

Use Python 3.12 (supported range: 3.11–3.13). Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) if necessary:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd nanoclip
uv sync --extra cpu
```

The `cpu` extra selects CPU wheels on Linux and the normal PyTorch wheel on
macOS, which also supports MPS. Development tools are included by default.
`uv sync` creates `.venv` and resolves `uv.lock`; keep that lockfile with your
experiment to reproduce its dependencies. The commands below use `--no-sync`
to preserve the explicitly selected CPU or GPU environment.

```bash
uv run --no-sync bash scripts/check.sh  # lint, formatting, all tests
uv run --no-sync python -m pytest -q  # tests only; CUDA tests skip without a GPU
bash runs/smoke.sh                    # 10 CPU steps, validation, latest.pt
```

For Apple Silicon, use `--device mps`, or let `--device auto` choose CUDA, MPS,
then CPU. CPU and MPS training use fp32 in this iteration.

## Train on a small dataset

The smoke dataset renders colored shapes with descriptive captions on demand;
it is deterministic and requires no download. To exercise the complete local
image/JSONL pipeline, first save a small shape dataset:

```bash
uv run --no-sync python -m scripts.prepare_data \
  --source shapes --samples 256 --output data/shapes
uv run --no-sync python -m scripts.train \
  --size tiny --data data/shapes/train.jsonl --val-data data/shapes/val.jsonl \
  --batch-size 16 --steps 200 --output-dir checkpoints/shapes
```

For natural images, the optional importer streams a subset of
[Flickr8k](https://huggingface.co/datasets/jxie/flickr8k). It takes the first N
training images, uses `caption_0` once per image, and creates a seeded 80/20
experiment split. This is a small practice split, not a standard benchmark split.
The upstream dataset's terms apply.

```bash
uv sync --extra cpu --extra data
uv run --no-sync python -m scripts.prepare_data \
  --source flickr8k --samples 512 --output data/flickr8k
uv run --no-sync python -m scripts.train \
  --size tiny --data data/flickr8k/train.jsonl --val-data data/flickr8k/val.jsonl \
  --batch-size 32 --steps 1000 --output-dir checkpoints/flickr8k
```

Bring your own images by making two JSONL manifests, one record per line:

```json
{"image": "images/001.jpg", "caption": "a dog running through grass"}
{"image": "images/002.jpg", "caption": "a red boat on a lake"}
```

Paths are relative to the manifest (absolute paths also work). Split by image so
an image's captions stay in one split. Prefer one caption per image during training:
the diagonal loss treats every other record as a negative, including duplicates.
For evaluation, an optional `"image_id"` groups multiple captions; its default
is the resolved image path. Retrieval accepts any caption of the matching image
and keeps one gallery image per ID.

## Training and checkpoints

```bash
uv run --no-sync python -m scripts.train \
  --size tiny --device cpu --batch-size 8 --grad-accum 2 \
  --samples 128 --val-samples 32 --steps 100 --output-dir checkpoints/example
uv run --no-sync python -m scripts.train \
  --resume checkpoints/example/latest.pt --device cpu --steps 200
```

`--steps` is the total target number of optimizer-step attempts, including steps
already saved. Resume restores unspecified training options, model configuration,
AdamW state, AMP scaler, RNG state, epoch, and batch cursor. Batch size,
accumulation, seed, optimizer settings, and model size must stay the same; start
a fresh run to change these. Device, precision, normalization backend, worker
count, output location, and data paths can change. Training manifest contents
must remain identical; relative paths make moving the dataset straightforward.
Bitwise equivalence is tested on CPU with the same environment; switching device
or precision need not reproduce identical floating-point results.

The run directory contains `latest.pt`, `config.json`, and append-only
`metrics.jsonl`. Checkpoints are atomically replaced after complete optimization
steps, every `--save-every` steps and at normal completion. A sudden instance stop
loses work since the last saved step. A fresh run refuses to overwrite an existing
checkpoint. Logs include contrastive loss, temperature scale, and validation
recall. An existing metrics file may include trailing records newer than a resumed
checkpoint after a crash; use a new `--output-dir` for an unambiguous resumed log.

`--batch-size` is the number of contrastive pairs **in each microbatch**.
Accumulation averages gradients across `--grad-accum` microbatches; it does not
share negatives between them. Thus `batch=16, accumulation=4` has 16 candidates
per contrastive task, unlike `batch=64`. The final accumulation window of an
epoch uses its actual number of microbatches. Training drops the last incomplete
batch; validation keeps every example for retrieval and omits singleton batches
only from the loss. Validation loss depends on its batch size. Tiny synthetic
datasets can contain repeated scenes/captions, so their retrieval scores are
diagnostics, not evidence of semantic generalization.

CUDA supports `--precision fp32|fp16|bf16|auto`. Auto selects bf16 when supported,
otherwise fp16; fp16 uses a gradient scaler. AdamW uses a constant learning rate
and gradient clipping at norm 1. There is no distributed training or scheduler.

## Retrieval and zero-shot example

```bash
uv run --no-sync python -m scripts.evaluate \
  --checkpoint checkpoints/shapes/latest.pt --data data/shapes/val.jsonl
uv run --no-sync python -m scripts.zero_shot \
  --checkpoint checkpoints/shapes/latest.pt --image data/shapes/images/000000.png \
  --template '{}' --labels 'a red circle' 'a blue square' 'a green triangle'
```

Evaluation reports image-to-text and text-to-image Recall@1/5/10 against the
entire held-out gallery, using chunks to bound the similarity matrix's memory.
Embeddings are collected on CPU. `k` is capped at gallery size. Zero-shot inference
ranks text prompts without a classifier head; scores are a softmax over the
provided labels, not calibrated probabilities. Useful predictions depend on the
training data and training time. The default prompt template is `a photo of {}`.

## Single cloud GPU: RunPod or Vast.ai

Use one RTX 3090, RTX 4090, or A10 with Linux x86_64, Python 3.12, a C compiler,
and a current host NVIDIA driver compatible with CUDA 12.8. Choose a provider
CUDA/PyTorch template with GPU passthrough already configured. `nvidia-smi` must
work inside the instance. PyTorch 2.9.1 has an official CUDA 12.8 wheel; the GPU
extra selects that build and Triton 3.5.1. See the
[PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)
and [NVIDIA driver compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The wheel provides the CUDA runtime; you do not need to build PyTorch or install
a separate full CUDA toolkit for these scripts. The host driver is configured
by the provider, not by `uv`.

Copy this repository onto the instance, then:

```bash
# Ubuntu container, if compiler/Python headers are missing (run as root):
apt-get update
apt-get install -y build-essential python3-dev
# Install uv using the earlier command, then from this repository:
nvidia-smi
uv sync --extra gpu
uv run --no-sync python -c "import torch, triton; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.get_device_name())"
uv run --no-sync python -m pytest -q -m cuda
```

Put both data and checkpoints on persistent storage. For RunPod, attach a
**network volume** and use its `/workspace` mount. Container disk is ephemeral;
a Pod volume does not have the same lifetime as a network volume. Consult
[RunPod's storage types](https://docs.runpod.io/pods/storage/types).
For Vast.ai, attach a persistent volume, normally mounted at `/data`; volumes
are tied to a physical machine, so a different host requires transferring data.
See [Vast.ai volumes](https://docs.vast.ai/guides/instances/storage/volumes).

```bash
export PERSIST_ROOT=/workspace/nanoclip-data   # RunPod network volume
# On Vast.ai instead: export PERSIST_ROOT=/data/nanoclip-data
export DATA_ROOT="$PERSIST_ROOT/data/shapes"
export RUN_DIR="$PERSIST_ROOT/runs/tiny"
mkdir -p "$PERSIST_ROOT"
uv run --no-sync python -m scripts.prepare_data \
  --source shapes --samples 512 --output "$DATA_ROOT"
STEPS=2000 BATCH_SIZE=64 bash runs/train_gpu.sh
```

The runner defaults to the PyTorch normalization baseline and bf16. To exercise
Triton, use a fresh run directory:

```bash
RUN_DIR="$PERSIST_ROOT/runs/tiny-triton" NORM=triton bash runs/train_gpu.sh
```

`SIZE=small`, `BATCH_SIZE`, `GRAD_ACCUM`, `PRECISION`, `WORKERS`, and `STEPS` are
environment overrides. Start with `tiny`; reduce batch size for a fresh run if
memory is tight. Measured memory and throughput depend on the GPU and precision.
For Flickr8k on the GPU, install with `uv sync --extra gpu --extra data` before
running the importer. A `tmux`/`screen` session keeps training alive when SSH
disconnects; it does not protect against instance termination.

After restart, reinstall the environment if needed, reattach the volume, set
`DATA_ROOT` and `RUN_DIR`, and resume directly to inherit the saved training settings:

```bash
uv sync --extra gpu
uv run --no-sync python -m scripts.train \
  --resume "$RUN_DIR/latest.pt" --device cuda --steps 4000 \
  --data "$DATA_ROOT/train.jsonl" --val-data "$DATA_ROOT/val.jsonl"
```

To move instances, stop the training process after a checkpoint, copy the run
directory and the complete dataset tree (including manifests), and run the same
resume command with the new paths. For example, after replacing the SSH host
and destination with your instance's values:

```bash
rsync -avP "$PERSIST_ROOT/" user@new-host:/data/nanoclip-data/
```

Copy the source and resolved `uv.lock` too; recreate `.venv` on the destination.
Keep an independent backup before deleting an instance or volume.

## The first Triton lesson: L2 normalization

`nanoclip/normalization.py` is the backend boundary. The reference computes
`y = x / max(norm(x), eps)`, with fp32 accumulation for fp16/bf16 inputs and
the original dtype at the output. The Triton implementation assigns one program
to each row, masks a power-of-two block, and performs its reduction in fp32.

With upstream gradient `g`, `r = norm(x)`, and `d = max(r, eps)`:

```text
r >= eps: dx = (g - y * sum(g * y)) / d
r <  eps: dx = g / eps
```

The equality convention matches PyTorch's `clamp_min`. Backward saves the input
and inverse norm to reconstruct `y` in fp32, avoiding errors from saved rounded
half-precision outputs. A `torch.autograd.Function` launches both kernels.
Noncontiguous input/gradient tensors are copied explicitly. This first kernel
supports nonempty 2-D fp32/fp16/bf16 tensors with up to 16,384 columns and
first-order gradients; higher-order derivatives use the PyTorch backend.
An explicit Triton request on an unsupported device raises an error.

```bash
uv run --no-sync python -m pytest -q tests/test_triton_normalization.py
uv run --no-sync python -m benchmarks.bench_l2_norm \
  --rows 256 4096 --dims 64 128 512 --dtype fp32 --mode both
```

The benchmark warms up compilation, checks outputs, and reports GPU timings for
forward and backward separately. Backward timing retains a forward graph;
forward timing disables autograd. Wrapper/allocation overhead is included.
There is no claimed speedup: for small embeddings launch overhead can dominate,
and this operation may be only a small part of training time. Measure complete
training before drawing conclusions. The
[Triton reduction/backward tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
is useful background for extending the experiment.

Future experiments can add softmax, LayerNorm, and matrix multiplication under
`nanoclip/kernels/`, each with a PyTorch reference, forward/backward comparisons,
and its own benchmark. Keep the reference path working throughout.

## Repository map

```text
nanoclip/     model, encoders, tokenizer, loss, datasets, runtime, evaluation, checkpoints
  kernels/    optional Triton forward/backward kernels
scripts/      train, evaluate, zero_shot, prepare_data, check.sh
runs/         CPU smoke test and single-GPU example commands
tests/        model/loss/data tests, resume regression, CUDA kernel comparisons
benchmarks/   L2 forward and backward GPU experiments
```

Tests cover both public model sizes, gradients into both towers and temperature,
loss against a manual reference, overfitting a paired batch, byte tokenization,
padding invariance, normalization gradcheck, JSONL loading, full-gallery retrieval,
and exact CPU checkpoint/resume across epoch boundaries. CUDA tests compare outputs
and gradients for multiple sizes and dtypes, noncontiguous layouts, zero and small
norms, the epsilon boundary, and a complete model backward pass. They skip cleanly
without CUDA/Triton; a CPU test pass does not validate GPU kernel execution.

The CLIP objective follows [Radford et al. (2021)](https://arxiv.org/abs/2103.00020).
