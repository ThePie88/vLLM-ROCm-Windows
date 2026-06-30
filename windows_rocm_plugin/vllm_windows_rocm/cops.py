"""Fallback implementations of the vLLM `torch.ops._C.*` fused ops, for the no-native-kernel
(VLLM_TARGET_DEVICE=empty) Windows build.

vLLM's CustomOp layers bind `torch.ops._C.<op>` in __init__/forward_cuda (e.g.
SiluAndMul.__init__ does `self.op = torch.ops._C.silu_and_mul` unconditionally on cuda-alike),
so on an empty build they crash before any forward_native fallback. We register the `_C`
op namespace here with correct schemas (mirrored from csrc/torch_bindings.cpp) and torch-native
implementations, so the model runs end-to-end.

These are CORRECTNESS-first (torch ops). Phase 2 replaces the hot ones with fused Triton/HIP
kernels for speed — same op names, so no vLLM changes needed.

Native-kernel handoff: if the compiled `vllm_win_C` library (built from vLLM's own csrc via
experiments/vllm_c_ext/) is present, we load it FIRST. Its TORCH_LIBRARY(_C) registrations
(silu_and_mul/rms_norm/fused_add_rms_norm/rotary_embedding, real HIP kernels) then take over,
and the fallbacks below are installed only for ops the native lib does NOT provide (e.g.
weak_ref_tensor). Set VLLM_WIN_C_DIR to override the build dir; set VLLM_WIN_C_NATIVE=0 to
force pure fallbacks (for A/B measurement).
"""
import glob
import os
import sys

import torch

_INSTALLED = False
_NATIVE_DIR = os.environ.get("VLLM_WIN_C_DIR", r"C:\vw_cext_build")


def _load_native() -> str | None:
    """Load the compiled vLLM _C kernels (vllm_win_C.pyd) so its TORCH_LIBRARY(_C) wins."""
    if os.environ.get("VLLM_WIN_C_NATIVE", "1") == "0":
        return None
    cands = sorted(glob.glob(os.path.join(_NATIVE_DIR, "vllm_win_C*.pyd")))
    for p in cands:
        # torch.ops.load_library binds the TORCH_LIBRARY static initializers without needing
        # a Python import; dependent DLLs (c10/torch_hip/amdhip64) are already in-process.
        try:
            torch.ops.load_library(p)
            return p
        except Exception as e:
            print("vllm-win native _C load_library warning:", repr(e))
            # fallback: import as a module (adds nothing extra, but uses Python's loader path)
            try:
                d = os.path.dirname(p)
                if d not in sys.path:
                    sys.path.insert(0, d)
                import importlib
                importlib.import_module(os.path.splitext(os.path.basename(p))[0])
                return p
            except Exception as e2:
                print("vllm-win native _C import warning:", repr(e2))
    return None


def _silu_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def _gelu_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.gelu(x[..., :d]) * x[..., d:])


def _gelu_tanh_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    # gemma uses GeGLU with the tanh GELU approximation.
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.gelu(x[..., :d], approximate="tanh") * x[..., d:])


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


def _moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids,
                          experts_ids, num_tokens_post_pad, maybe_expert_map):
    # Torch fallback for torch.ops._moe_C.moe_align_block_size (no _moe_C on Windows).
    # Groups the flattened top-k slot indices by expert, pads each expert's run up to a
    # multiple of block_size, and writes: sorted_token_ids (slot indices, padding=numel
    # sentinel), experts_ids (expert per block, -1 past the end), num_tokens_post_pad.
    # CUDAGRAPH-SAFE: fixed output shapes, no .item()/host sync, no data-dependent sizes,
    # so the FULL_DECODE_ONLY graph can capture the MoE decode path.
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.long)
    numel = flat.numel()
    sorted_token_ids.fill_(numel)  # sentinel = num valid slots (masked by the gemm kernel)
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    counts.scatter_add_(0, flat.clamp(0, num_experts - 1), torch.ones_like(flat))
    padded = ((counts + block_size - 1) // block_size) * block_size
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(padded, 0)
    # scatter each slot to its padded destination (fully vectorized, fixed-size)
    order = torch.argsort(flat, stable=True)            # slot indices grouped by expert
    sorted_experts = flat[order]
    cnt_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    cnt_offsets[1:] = torch.cumsum(counts, 0)
    ranks = torch.arange(numel, device=device) - cnt_offsets[sorted_experts]
    dst = offsets[sorted_experts] + ranks
    sorted_token_ids[dst] = order.to(sorted_token_ids.dtype)
    # expert id per block over the FULL preallocated experts_ids (fixed size); -1 past total
    nb = experts_ids.shape[0]
    block_starts = torch.arange(nb, device=device) * block_size
    total = offsets[num_experts]
    eids = (torch.searchsorted(offsets, block_starts, right=True) - 1).clamp(0, num_experts - 1)
    eids = torch.where(block_starts < total, eids, torch.full_like(eids, -1))
    if maybe_expert_map is not None:
        mapped = maybe_expert_map[eids.clamp(min=0)]
        eids = torch.where(eids >= 0, mapped, torch.full_like(eids, -1))
    experts_ids.copy_(eids.to(experts_ids.dtype))
    num_tokens_post_pad.copy_(total.reshape(num_tokens_post_pad.shape).to(num_tokens_post_pad.dtype))


def _moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    # output[T, H] = sum over the top_k dim of input[T, top_k, H] (fp32 accumulate).
    output.copy_(input.to(torch.float32).sum(dim=1).to(output.dtype))


def _install_moe_C() -> None:
    """Load native `_moe_C` (built from csrc/moe via build_moe_c.py) if present; otherwise
    register torch fallbacks for the fused-MoE ops the standard fused_experts path needs."""
    # Native first: its TORCH_LIBRARY(_moe_C) (moe_align_block_size/moe_sum/topk_softmax/
    # batched_moe_align_block_size, real HIP kernels) wins over the torch fallbacks below.
    if os.environ.get("VLLM_WIN_MOE_NATIVE", "1") != "0":
        moe_dir = os.environ.get("VLLM_WIN_MOE_DIR", r"C:\vw_moe_build")
        for p in sorted(glob.glob(os.path.join(moe_dir, "vllm_win_moe_C*.pyd"))):
            try:
                torch.ops.load_library(p)
                print("vllm-win: loaded native _moe_C from", p)
                break
            except Exception as e:  # noqa: BLE001
                print("vllm-win native _moe_C load warning:", repr(e))
    if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "moe_align_block_size"):
        return  # native present -> use it, skip torch fallbacks
    lib = torch.library.Library("_moe_C", "FRAGMENT")
    ops = [
        ("moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
         "Tensor(a!) sorted_token_ids, Tensor(b!) experts_ids, Tensor(c!) num_tokens_post_pad, "
         "Tensor? maybe_expert_map) -> ()", _moe_align_block_size),
        ("moe_sum(Tensor input, Tensor(a!) output) -> ()", _moe_sum),
    ]
    for schema, fn in ops:
        name = schema.split("(", 1)[0]
        try:
            lib.define(schema)
        except Exception:
            pass
        for key in ("CUDA", "CPU"):
            try:
                lib.impl(name, fn, key)
            except Exception:
                pass
    globals()["_MOE_LIB"] = lib


def _unsupported_op(*args, **kwargs):
    raise NotImplementedError(
        "this fused fp8/fp4 _C op is not implemented on the Windows ROCm build. It is "
        "registered only as a stub so vLLM's inductor fusion-pass matchers (which build "
        "torch.ops._C.<op>.default dicts at import time) can load. It is never called in "
        "eager / CompilationMode.NONE."
    )


# Stub schemas (mirrored from csrc/torch_bindings.cpp) for the fused fp8/fp4 quant ops that
# vLLM's fusion-pass matchers reference at module-import time (act_quant_fusion / rms_quant_fusion
# / qk_norm_rope_fusion via matcher_utils, imported for any cuda_alike platform incl. ROCm).
# Defining them makes `torch.ops._C.<op>.default` resolvable so model init (which imports the
# pass manager) succeeds even though no native _C kernels exist. The impl raises if ever called.
_STUB_OPS = [
    "static_scaled_fp8_quant(Tensor(a!) result, Tensor input, Tensor scale, (int, int)? group_shape=None) -> ()",
    "dynamic_scaled_fp8_quant(Tensor(a!) result, Tensor input, Tensor(b!) scale) -> ()",
    "dynamic_per_token_scaled_fp8_quant(Tensor(a!) result, Tensor input, Tensor(b!) scale, Tensor? scale_ub) -> ()",
    "scaled_fp4_quant(Tensor input, Tensor input_scale, bool is_sf_swizzled_layout) -> (Tensor, Tensor)",
    # vLLM's _custom_ops.py register_fake's BOTH scaled_fp4_quant and its .out overload (guarded
    # by hasattr(_C, "scaled_fp4_quant")); defining the base flips that guard True, so the .out
    # overload must exist too.
    ("scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool is_sf_swizzled_layout, *, "
     "Tensor(a!) output, Tensor(b!) output_scale) -> ()"),
    ("per_token_group_fp8_quant(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, int group_size, "
     "float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool dummy_is_scale_transposed, "
     "bool dummy_is_tma_aligned) -> ()"),
    "rms_norm_static_fp8_quant(Tensor(a!) result, Tensor input, Tensor weight, Tensor scale, float epsilon) -> ()",
    ("fused_add_rms_norm_static_fp8_quant(Tensor(a!) result, Tensor input, Tensor(b!) residual, Tensor weight, "
     "Tensor scale, float epsilon) -> ()"),
    ("rms_norm_dynamic_per_token_quant(Tensor(a!) result, Tensor input, Tensor weight, Tensor(b!) scale, "
     "float epsilon, Tensor? scale_ub, Tensor(c!)? residual) -> ()"),
    ("rms_norm_per_block_quant(Tensor(a!) result, Tensor input, Tensor weight, Tensor(b!) scale, float epsilon, "
     "Tensor? scale_ub, Tensor(c!)? residual, int group_size, bool is_scale_transposed) -> ()"),
    ("fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, "
     "float eps, Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, bool is_neox, Tensor position_ids) -> ()"),
    "silu_and_mul_quant(Tensor(a!) result, Tensor input, Tensor scale) -> ()",
    ("silu_and_mul_nvfp4_quant(Tensor(a!) result, Tensor(b!) result_block_scale, Tensor input, "
     "Tensor input_global_scale) -> ()"),
]


_OPS = [
    ("silu_and_mul(Tensor(a!) result, Tensor input) -> ()", _silu_and_mul),
    ("gelu_and_mul(Tensor(a!) result, Tensor input) -> ()", _gelu_and_mul),
    ("gelu_tanh_and_mul(Tensor(a!) result, Tensor input) -> ()", _gelu_tanh_and_mul),
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
    # Load the compiled native kernels first; their TORCH_LIBRARY(_C) ops win over fallbacks.
    native = _load_native()
    lib = torch.library.Library("_C", "FRAGMENT")
    for schema, fn in _OPS:
        name = schema.split("(", 1)[0]
        # Per-op guard: if the native lib already provides this op, don't shadow it (but DO
        # still register the others, e.g. weak_ref_tensor, which the native lib lacks).
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, name):
            continue
        try:
            lib.define(schema)
        except Exception:
            pass  # already defined
        for key in ("CUDA", "CPU", "Meta"):
            try:
                lib.impl(name, fn, key)
            except Exception:
                pass
    # Stub-register the fused fp8/fp4 quant ops so the fusion-pass matchers can import.
    for schema in _STUB_OPS:
        name = schema.split("(", 1)[0]
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, name):
            continue
        try:
            lib.define(schema)
        except Exception as e:  # noqa: BLE001
            print("vllm-win cops stub define warning:", name, repr(e))
            continue
        # Only register CUDA: vLLM adds its own register_fake (Meta) for the tensor-returning
        # ops (e.g. scaled_fp4_quant); registering Meta here would collide. CUDA-only means an
        # accidental eager call raises our clear NotImplementedError instead of silent garbage.
        try:
            lib.impl(name, _unsupported_op, "CUDA")
        except Exception:
            pass
    # keep a ref so the Library isn't GC'd
    globals()["_LIB"] = lib
    _install_moe_C()  # fused-MoE ops (_moe_C namespace) torch fallbacks
    _INSTALLED = True
    if native:
        present = [s.split("(", 1)[0] for s, _ in _OPS
                   if hasattr(torch.ops._C, s.split("(", 1)[0])]
        print("vllm-win: native _C kernels loaded from", native, "| ops:", present)
