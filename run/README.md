# Phase 1 — vLLM first token on native Windows + ROCm (gfx1100)

**Status: PASSED (2026-06-30)** on AMD Radeon RX 7900 XT (gfx1100), Windows 11 — believed world-first.

```
PROMPT: 'Hello, my name is'
OUTPUT: ' J.C. and I am a student at the University of California, Berkeley'
```
OPT-125m, eager mode, `TRITON_ATTN`, single GPU, no custom kernels. KV cache 274,320 tokens / 9.42 GiB.

## How it's wired

1. **vLLM source** — `../vllm` is vLLM **v0.19.1** (pins torch 2.10.0 + torchvision 0.25.0, matching the installed ROCm stack; v0.20+ wants torch 2.11). Installed editable with **no kernels**:
   ```bash
   cd ../vllm
   python use_existing_torch.py                     # strip torch pins so pip won't replace the ROCm build
   VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
   ```
2. **Out-of-tree plugin** — `../windows_rocm_plugin` (`pip install -e .`). Provides:
   - `WindowsRocmPlatform` (registered via the `vllm.platform_plugins` entry point) — overrides the amdsmi-based device methods to use `torch.cuda`.
   - `torchdist_shim` — a **single-process `torch.distributed`** (this torch is built USE_DISTRIBUTED=0) plus Windows stubs for `amdsmi`, `uvloop`, `fcntl`, and `torch._C._distributed_c10d`.
3. **One-line vendored patch** — `../vllm/vllm/__init__.py` imports `vllm_windows_rocm.bootstrap` (applies the shim before any vllm submodule loads torch.distributed).
4. **Extra deps** — `pip install llguidance xgrammar` (structured-output backends; Windows wheels exist).

## Running

```bash
# IMPORTANT: run from this run/ dir (NOT the project root), else the vllm/ clone dir
# shadows the installed vllm package as a namespace package.
python first_token.py
```

Key runtime config (see `first_token.py`): `enforce_eager=True`, `attention_backend="TRITON_ATTN"`
(uses Triton reshape_and_cache + attention, avoiding the missing `_rocm_C`/`_C_cache_ops` kernels;
ROCM_ATTN is the default but needs C kernels), `tensor_parallel_size=1`, and env
`VLLM_ROCM_USE_SKINNY_GEMM=0` (→ `torch.nn.functional.linear` instead of `_rocm_C.wvSplitK`),
`VLLM_ROCM_USE_AITER=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

> NOTE: in v0.19.1 the `VLLM_ATTENTION_BACKEND` env var is gone — use the `attention_backend=` kwarg.

## What this proves / what's next
Proves the **full vLLM engine** (scheduler, KV-cache, paged attention via Triton, model runner,
sampler) runs on native Windows + ROCm gfx1100 in eager mode. This is correctness, not speed
(~4–5 tok/s on a tiny model in eager). Phase 2 = real paged-attention performance (Triton tuning /
hand-written WMMA kernels — proven feasible in `../experiments/`), then W4A16 quant, then KVarN.
