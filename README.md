# vLLM on native Windows + AMD ROCm (RDNA3)

Glue and build tooling to run [vLLM](https://github.com/vllm-project/vllm) on **native
Windows** (no WSL2) with **AMD ROCm** on **RDNA3** consumer GPUs. Developed and tested on a
**Radeon RX 7900 XT (gfx1100)**.

This is **not** a fork of vLLM. It is an out-of-tree platform plugin plus a set of
compatibility shims, a one-line patch, and a build harness. Upstream vLLM is cloned and
pinned separately (see Setup).

## Status (honest)

Experimental / early. What currently works on the test machine:

- vLLM imports and **generates correct tokens** on gfx1100, native Windows, single GPU.
- Weight-quantized models run: **compressed-tensors / AWQ W4A16** via the pure-Triton
  `conch` kernel (no CUDA-only Marlin needed).
- A real 9B Qwen3.5 hybrid (linear + full attention) MoE model loads and runs.
- **hipGraph capture works** (`cudagraph_mode=FULL_DECODE_ONLY`, no inductor).

What is **not** done:

- **Performance is not yet competitive.** Single-stream decode on the 9B hybrid is around
  ~12 tok/s, versus ~30 tok/s for a comparable GGUF on llama.cpp/Vulkan on the same card.
  The decode is kernel-bound (unfused norms/activations as torch fallbacks, the Triton GDN
  linear-attention path, MoE). Porting vLLM's native HIP kernels (`csrc/`) to Windows and
  fusing the hot paths is in progress.
- **Single GPU only.** RCCL does not exist on Windows, so tensor/pipeline parallel are out
  of scope; `torch.distributed` is shimmed for the single-process case only.
- Large-context VRAM: KV-cache quantization (INT8 / 2-bit) is not wired up yet.

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
  - `cops.py`: torch-native fallbacks for the `torch.ops._C.*` fused ops that vLLM binds
    unconditionally (`silu_and_mul`, `rms_norm`, `fused_add_rms_norm`, `rotary_embedding`,
    `weak_ref_tensor`). These are correctness-first; replacing them with native HIP kernels
    is the current work.
- vLLM is installed with `VLLM_TARGET_DEVICE=empty` (no kernels compiled) plus a one-line
  patch to `vllm/__init__.py` that imports the shim early.

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
```

## Running

Run from `run/` (not the repo root, so the cloned `vllm/` directory does not shadow the
installed `vllm` package).

```bat
cd run
python first_token.py        :: smallest end-to-end smoke test (OPT-125m)
python bench.py              :: throughput + VRAM (configure via VLLM_BENCH_* env vars)
```

For a quantized model with a broken tokenizer_class (e.g. some llm-compressor exports):

```bat
python ..\tools\fix_tokenizer_config.py <model-substring>
set HF_HUB_OFFLINE=1
```

## Layout

- `windows_rocm_plugin/` - the out-of-tree platform plugin and compatibility shims
- `tools/` - patch and fixup scripts
- `run/` - bench / first-token / profiling drivers
- `experiments/` - standalone HIP/Triton kernel proofs (build harness, rocWMMA, etc.)
- `research/` - design notes (Italian)

## License

This repository's glue code is Apache-2.0, matching vLLM. vLLM itself is not included here
and remains under its own license.
