"""Fallback implementations of the vLLM `torch.ops._C.*` fused ops, for the no-native-kernel
(VLLM_TARGET_DEVICE=empty) Windows build.

vLLM's CustomOp layers bind `torch.ops._C.<op>` in __init__/forward_cuda (e.g.
SiluAndMul.__init__ does `self.op = torch.ops._C.silu_and_mul` unconditionally on cuda-alike),
so on an empty build they crash before any forward_native fallback. We register the `_C`
op namespace here with correct schemas (mirrored from csrc/torch_bindings.cpp) and torch-native
implementations, so the model runs end-to-end.

These are CORRECTNESS-first (torch ops). Phase 2 replaces the hot ones with fused Triton/HIP
kernels for speed — same op names, so no vLLM changes needed.
"""
import torch

_INSTALLED = False


def _silu_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def _rms_norm(result: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    result.copy_(xf.to(orig) * weight)


def _fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                        eps: float) -> None:
    orig = x.dtype
    added = x + residual
    xf = added.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    out = xf.to(orig) * weight
    x.copy_(out)
    residual.copy_(added)


def _weak_ref_tensor(x: torch.Tensor) -> torch.Tensor:
    # vLLM's CUDA-graph path uses this to hold output buffers without bumping the storage
    # refcount. A view aliasing the same storage is a correct (if slightly less weak) fallback.
    return x.view(x.shape)


def _rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox) -> None:
    # Reuse vLLM's own native RoPE math (handles neox/gptj + partial rotary correctly).
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rotary_dim = cos_sin_cache.shape[-1]
    q2, k2 = RotaryEmbedding.forward_static(
        positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox
    )
    query.copy_(q2)
    if key is not None and k2 is not None:
        key.copy_(k2)


_OPS = [
    ("silu_and_mul(Tensor(a!) result, Tensor input) -> ()", _silu_and_mul),
    ("rms_norm(Tensor(a!) result, Tensor input, Tensor weight, float epsilon) -> ()", _rms_norm),
    ("fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float epsilon) -> ()",
     _fused_add_rms_norm),
    ("rotary_embedding(Tensor positions, Tensor(a!) query, Tensor(b!)? key, int head_size, "
     "Tensor cos_sin_cache, bool is_neox) -> ()", _rotary_embedding),
    ("weak_ref_tensor(Tensor(a) input) -> Tensor(a)", _weak_ref_tensor),
]


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    # If a real _C with these ops exists (a future native build), don't shadow it.
    if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "silu_and_mul"):
        _INSTALLED = True
        return
    lib = torch.library.Library("_C", "FRAGMENT")
    for schema, fn in _OPS:
        name = schema.split("(", 1)[0]
        try:
            lib.define(schema)
        except Exception:
            pass  # already defined
        for key in ("CUDA", "CPU", "Meta"):
            try:
                lib.impl(name, fn, key)
            except Exception:
                pass
    # keep a ref so the Library isn't GC'd
    globals()["_LIB"] = lib
    _INSTALLED = True
