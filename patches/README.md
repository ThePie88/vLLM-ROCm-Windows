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
