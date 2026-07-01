# vLLM clone patches

The `vllm/` directory in this repo is a clone of upstream vLLM and is **gitignored** (it is the
build/runtime source tree, not part of this repo's history). A few fixes for the native
Windows + ROCm port are made as **direct edits to that clone**, so they are captured here as
patches for reproducibility. Everything else (the platform plugin, native-kernel builds, run
harness) lives in this repo and is monkeypatched/loaded at runtime without touching vLLM source.

Clone base when these were generated: vLLM `b1388b1` (v0.19.2.dev0).

Apply from the repo root:

```
git -C vllm apply ../patches/vllm/conch-group-size.patch
git -C vllm apply ../patches/vllm/gemma4-moe-weightload.patch
git -C vllm apply ../patches/vllm/kvarn.patch
```

## conch-group-size.patch
`conch.py`: conch's Triton W4A16 kernel applies one scale per `block_k == 64` tile, so it is only
numerically correct for `group_size >= 64`. Verified: gs 128/64 give rel_err ~3e-3, gs 32 gives
~0.96 (garbage -> degenerate output). Reverts the whitelist to `[-1, 64, 128]` (drops the wrong
32). group_size 32 (e.g. gemma4 AWQ) is instead routed to `WinRocmW4A16DequantKernel` in the
plugin (a correct dequant->matmul / fused GEMV fallback).

## gemma4-moe-weightload.patch
`gemma4.py` `_weight_iterator`: the fused-3D-expert explosion kept the checkpoint's underscore
quant suffix (`gate_proj_packed`/`_scale`), which never matched `expert_params_mapping`
(`experts.{id}.{proj}.` dotted) -> `KeyError 'layers.0.moe.experts.0.down_proj_packed'`. Rewrites
`_packed`/`_scale` to the canonical dotted `.weight_packed`/`.weight_scale` so compressed-tensors
fused-expert MoE checkpoints load 1:1 with the Linux behavior. Regression-safe: bare/`.weight`
(unquantized) names are untouched.

## kvarn.patch
Port of Huawei **KVarN** (calibration-free KV-cache quant: Hadamard rotation + Sinkhorn
variance-normalisation + asymmetric RTN, K 4-bit per-channel / V 2-or-4-bit per-token, per
128-token tile) as a native vLLM KV-cache-dtype backend on Windows + ROCm (gfx1100). Enable with
`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`.

Contents (one self-contained patch; new files + integration edits):
- **New files** copied from github.com/huawei-csl/KVarN (Apache 2.0) with two ROCm edits:
  `v1/attention/backends/kvarn_attn.py`, `v1/attention/ops/{kvarn_decode,kvarn_store,triton_kvarn_decode,triton_kvarn_sinkhorn}.py`,
  `model_executor/layers/quantization/kvarn/{__init__,config,sinkhorn}.py`. ROCm edits:
  (1) dropped the `maxnreg` autotune configs in `triton_kvarn_decode.py` (NVIDIA-only; Triton-AMDGPU
  raises "Keyword argument maxnreg unrecognised"), pinned to a single BLOCK_N=32/nw=4 config for fast
  first-run; (2) diagnostic env gates left inert-by-default in `kvarn_attn.py`
  (`KVARN_FORCE_SLOW` = dequant+SDPA reference path, `KVARN_NO_HADAMARD`, `KVARN_GTRACK`,
  `KVARN_RECON_DEBUG`, `KVARN_FAST_FLUSH=0` = legacy per-tile flush).
- **Integration edits** to vLLM: register the KVARN backend (`registry.py`); add the 4 kvarn presets
  to `CacheDType` (`config/cache.py`) and `STR_DTYPE_TO_TORCH_DTYPE` (`utils/torch_utils.py`); graft
  `TQFullAttentionSpec` (tile-quant full-attn spec with `tq_slot_size` byte sizing) into
  `v1/kv_cache_interface.py` and register it -> `FullAttentionManager` in
  `v1/core/single_type_kv_cache_manager.py`; `attention.py` `get_kv_cache_spec` returns
  `TQFullAttentionSpec` for `kvarn_*` layers (this branch takes PRECEDENCE over the sliding-window
  branch so `KVARN_QUANT_SLIDING` sliding layers get kvarn byte sizing, not fp16 SlidingWindowSpec).

THE key correctness fix lives OUTSIDE this patch, in `attention.py` too but as the plugin-critical
one-liner `self.impl.layer_name = prefix` (right after `self.impl = impl_cls(...)`): vLLM 0.19's
`Attention.__init__` never propagated the layer name to the impl, so KVarN's metadata builder found
zero impls for its group (`group_impls=0`) -> pool-slot allocation + tile flush never ran -> full
blocks were read back as uninitialised int4 (garbled decode). It is included in the attention.py hunk.

Status: correct end-to-end on gemma-4-26B (compressed-tensors W4A16 MoE); ~40 tok/s decode with
cudagraph (global-only) / real 4.4x KV-capacity win with `KVARN_QUANT_SLIDING=1` but slower pending a
builder D2H-sync refactor. Sliding-window semantics enforced by the decode kernel (`impl.sliding_window`).

## native-attn-sliding-window.patch
`csrc/attention/{attention_kernels.cuh, paged_attention_v1.cu}` + `csrc/ops.h`: threads a new
`sliding_window` int param through the generic wave32 paged_attention_v1 (device kernel -> global
kernel -> LAUNCH macro -> launcher -> public op -> ops.h decl). The QK stage gains an in-kernel
sliding-window mask `sw_mask = (sliding_window>0) && ((seq_len-1-token_idx) >= sliding_window)` ->
masked logit set to -FLT_MAX (excluded from the softmax normalizer AND the V accumulation), matching
vLLM's Triton `where((context_len-seq_offset) < SLIDING_WINDOW, S, -inf)`. `sliding_window<=0`
disables it (original full-attention behavior; perf unchanged). v2 is left intact (passes 0). This is
built (NOT compiled into the gitignored vllm tree at runtime) via `experiments/vllm_c_ext/build_attn_c.py`
into `vllm_win_attn_C.pyd` and loaded opt-in by the plugin (`VLLM_WIN_ATTN_NATIVE=1`).

Result: the native decode kernel is ~3.2x faster than Triton `kernel_paged_attention_2d` in isolation
and numerically correct, BUT the end-to-end ROCM_ATTN integration REGRESSES (-9% gemma, -5% ERNIE) --
the path overhead negates the kernel win. See `docs/s5-attention-lever-and-aiter-rdna3.md`. Kept
because the kernel is the RDNA3-native `fmha_v3` equivalent and the remaining flash-layout-swap path
would reuse it.
