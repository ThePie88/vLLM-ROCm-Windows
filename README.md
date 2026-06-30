# vLLM on native Windows + AMD ROCm (RDNA3)

Glue and build tooling to run [vLLM](https://github.com/vllm-project/vllm) on **native
Windows** (no WSL2) with **AMD ROCm** on **RDNA3** consumer GPUs. Developed and tested on a
**Radeon RX 7900 XT (gfx1100)**.

This is **not** a fork of vLLM. It is an out-of-tree platform plugin plus a set of
compatibility shims, a one-line patch, and a build harness that compiles vLLM's own HIP
kernels natively on Windows. Upstream vLLM is cloned and pinned separately (see Setup).

## Status (honest)

Experimental, but past "it just runs". What currently works on the test machine:

- vLLM imports and **generates correct tokens** on gfx1100, native Windows, single GPU.
- **compressed-tensors W4A16** models run, using vLLM's **native exllama GEMM** compiled for
  Windows (the pure-Triton `conch` kernel also works as a fallback).
- **torch.compile / inductor works** (CompilationMode.STOCK_TORCH_COMPILE), and **hipGraph
  decode capture works** (`cudagraph_mode=FULL_DECODE_ONLY`).
- Several of vLLM's own `csrc` HIP kernels are **compiled natively** and wired in (see below).

### Performance (measured)

Single-stream decode on the test machine, a 9B Qwen3.5 hybrid (linear + full attention)
model in compressed-tensors W4A16, batch 1, 8k context, greedy. Output was verified to be
identical across all three configurations.

| Configuration | decode |
| --- | --- |
| eager, torch fallbacks | 11.4 tok/s |
| + torch.compile (inductor) + hipGraph decode capture | 22.3 tok/s |
| + native W4A16 GEMM (exllama) | **39.9 tok/s** |

TTFT ~50 ms, ~17.7 GiB VRAM at this setting. Aggregate throughput scales with concurrency on
the same setup (greedy, 128 decode tokens): 73 tok/s at batch 4, 232 at batch 16, 358 at
batch 32.

Decode is still below the card's memory-bandwidth roofline; collapsing remaining per-step
host overhead, tuning the Triton linear-attention path, and porting the rest of the `csrc`
kernels are ongoing.

### Not done

- **Single GPU only.** RCCL does not exist on Windows, so tensor/pipeline parallel are out of
  scope; `torch.distributed` is shimmed for the single-process case only.
- Large-context VRAM: KV-cache quantization (INT8 / 2-bit) is not wired up yet.
- Only part of vLLM's kernel suite is built natively so far (see "Native kernels" below).

## Tested stack (pinned, fragile)

This depends on a specific, somewhat experimental combination. Other versions may not work.

- Windows 11, AMD Radeon RX 7900 XT (gfx1100)
- A ROCm-enabled PyTorch **Windows** build: `torch 2.10.0+rocm7.13` (TheRock-class), `torchvision 0.25.0`, Python 3.12
- AMD HIP SDK 7.2 (`C:\HIP-SDK`), MSVC (Visual Studio Build Tools), Windows SDK 10
- `triton-windows` 3.6, `conch-triton-kernels`, `llguidance`, `xgrammar`
- vLLM **v0.19.1** (the newest tag pinned to torch 2.10; v0.20+ requires torch 2.11)

Note: helper scripts contain absolute paths from the author's machine
(`C:\HIP-SDK`, `E:\BuildTools`, `C:\Users\...`). Adjust them for your environment.

## How it works

- `windows_rocm_plugin/` is a pip-installable package providing:
  - `WindowsRocmPlatform` (registered via the `vllm.platform_plugins` entry point) that
    detects the GPU through `torch.cuda` instead of the Linux-only `amdsmi`.
  - A **single-process `torch.distributed` shim** (the Windows ROCm torch wheel is built
    without distributed), plus stubs for `amdsmi`, `uvloop`, `fcntl`,
    `torch._C._distributed_c10d`, and a tokenizer-class compatibility alias.
  - A **`torch.distributed.tensor` stub** that makes the (natively absent) DTensor module
    raise `ModuleNotFoundError` instead of a half-initialized `ImportError`. inductor's graph
    logging guards that import with `except ModuleNotFoundError`; without the stub,
    `torch.compile` dies during compilation. This is what unblocks inductor here.
  - `cops.py`: loads the compiled native kernel library (see below) so `torch.ops._C.*`
    resolve to the real HIP kernels, and registers torch-native fallbacks for any op the
    native build does not provide (so vLLM's unconditional `torch.ops._C.*` bindings work
    either way).
- vLLM is installed with `VLLM_TARGET_DEVICE=empty` (no kernels compiled by vLLM's own build)
  plus a one-line patch to `vllm/__init__.py` that imports the shim early.

### Native kernels

`experiments/vllm_c_ext/` builds vLLM's **own** `csrc` HIP kernels for Windows. vLLM's
Linux build relies on a CUDA->HIP header redirect that the Windows torch wheel does not ship,
and `cpp_extension`'s hipify orchestrator mishandles Windows paths, so the harness applies
torch's hipify substitution engine (`RE_PYTORCH_PREPROCESSOR` + `PYTORCH_MAP`) to the sources
directly, with a small set of redirect shim headers. Currently built and validated:

- `silu_and_mul`, `rms_norm`, `fused_add_rms_norm`, `rotary_embedding` (fused activation /
  layernorm / RoPE)
- the **W4A16 GPTQ/exllama GEMM** (`gptq_gemm`, `gptq_shuffle`) from
  `csrc/quantization/gptq/q_gemm.cu`, which has a dedicated small-batch path for single-stream
  decode

To select the native exllama GEMM for a compressed-tensors W4A16 model, set
`VLLM_DISABLED_KERNELS=ConchLinearKernel` (vLLM's ROCm kernel selection then falls through
from `conch` to the exllama kernel).

## Setup

```bat
:: 1. Clone the matching vLLM tag next to this repo's content
git clone --depth 1 --branch v0.19.1 https://github.com/vllm-project/vllm.git vllm

:: 2. Don't let pip replace your ROCm torch, then install vLLM with no kernels
cd vllm
python use_existing_torch.py
set VLLM_TARGET_DEVICE=empty
python -m pip install -e . --no-build-isolation
cd ..

:: 3. Apply the one-line shim import to vLLM, install the plugin and extra deps
python tools\patch_vllm.py vllm
python -m pip install -e windows_rocm_plugin
python -m pip install conch-triton-kernels llguidance xgrammar

:: 4. (optional) Build vLLM's native HIP kernels for Windows
cd experiments\vllm_c_ext
build_run.bat
```

## Running

Run from `run/` (not the repo root, so the cloned `vllm/` directory does not shadow the
installed `vllm` package).

```bat
cd run
python first_token.py        :: smallest end-to-end smoke test (OPT-125m)
python bench.py              :: decode tok/s + VRAM (configure via VLLM_BENCH_* env vars)
python batch_sweep.py        :: aggregate throughput vs concurrency
```

`bench.py` knobs (env): `VLLM_BENCH_COMPILE=1` enables inductor, `VLLM_BENCH_CGMODE=FULL_DECODE_ONLY`
enables hipGraph decode capture, `VLLM_DISABLED_KERNELS=ConchLinearKernel` selects the native
exllama GEMM.

For a quantized model with a broken tokenizer_class (e.g. some llm-compressor exports):

```bat
python ..\tools\fix_tokenizer_config.py <model-substring>
set HF_HUB_OFFLINE=1
```

## Layout

- `windows_rocm_plugin/` - the out-of-tree platform plugin and compatibility shims
- `tools/` - patch and fixup scripts
- `run/` - bench / first-token / profiling / batch-sweep drivers
- `experiments/` - native `csrc` kernel build harness and standalone HIP/Triton kernel proofs

## License

This repository's glue code is Apache-2.0, matching vLLM. vLLM itself is not included here
and remains under its own license.
