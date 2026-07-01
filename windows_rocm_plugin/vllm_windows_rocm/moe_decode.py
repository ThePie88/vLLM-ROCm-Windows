"""Phase B: custom M=1 (decode) MoE path for compressed-tensors W4A16 on RDNA3.

At decode the batch is a single token, so only top_k (8) of the 128 experts are active. vLLM's
fused_experts builds the full sort/align/grouped-GEMM machinery (designed for throughput); for a
single token that is wasteful. This path instead loops the top_k active experts with a streaming
W4A16 dequant-GEMV in the moe_wna16 weight layout, bypassing the sort entirely.

Weight layout (post CompressedTensorsWNA16MoEMethod.process_weights_after_loading):
  w13_weight_packed [E, 2*I, H//2] uint8  (out-major, packed-along-K=H, 2 nibbles/byte)
  w13_weight_scale  [E, 2*I, H//G]
  w2_weight_packed  [E, H, I//2]  uint8
  w2_weight_scale   [E, H, I//G]
Symmetric int4 (bias 8). Gate/up are fused on dim 0 of w13 (first I = gate, next I = up).

Enable with VLLM_WIN_MOE_DECODE=1; VLLM_WIN_MOE_VALIDATE=1 compares vs fused_experts (eager).
"""
import os

import torch

from vllm.triton_utils import tl, triton

_BIAS = 8.0


@triton.jit
def _moe_gemv_batched_kernel(x_ptr, w_ptr, s_ptr, o_ptr, K, N, Kp, Gc, SXE,
                             BLOCK_N: tl.constexpr, GROUP: tl.constexpr, BIAS: tl.constexpr):
    # One launch over (E_active, N tiles): out[e, n] = sum_k x[e,k]*(nib(w[e,n,k])-BIAS)*s[e,n,k//G].
    # SXE = x expert-stride: 0 -> x shared across experts (gate_up); K -> per-expert x (down).
    e = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    HALF: tl.constexpr = GROUP // 2
    wrow = w_ptr + e * N * Kp + n[:, None] * Kp
    srow = s_ptr + e * N * Gc + n * Gc
    xrow = x_ptr + e * SXE
    num_groups = K // GROUP
    for g in range(num_groups):
        k0 = g * GROUP
        cols = (k0 // 2) + tl.arange(0, HALF)
        b = tl.load(wrow + cols[None, :], mask=nmask[:, None], other=0)
        lo = (b & 0xF).to(tl.float32) - BIAS
        hi = ((b >> 4) & 0xF).to(tl.float32) - BIAS
        xidx = k0 + tl.arange(0, HALF) * 2
        xlo = tl.load(xrow + xidx).to(tl.float32)
        xhi = tl.load(xrow + xidx + 1).to(tl.float32)
        contrib = tl.sum(lo * xlo[None, :] + hi * xhi[None, :], axis=1)
        s = tl.load(srow + g, mask=nmask, other=0.0).to(tl.float32)
        acc += contrib * s
    tl.store(o_ptr + e * N + n, acc.to(o_ptr.type.element_ty), mask=nmask)


def _moe_gemv_batched(x, w, s, group_size, x_per_expert):
    # w [E, N, K//2] uint8; s [E, N, K//G]; x [K] (shared) or [E, K] (per-expert) -> out [E, N]
    E, N, Kp = w.shape
    K = x.shape[-1]
    sxe = K if x_per_expert else 0
    o = torch.empty(E, N, dtype=x.dtype, device=x.device)
    grid = lambda m: (E, triton.cdiv(N, m["BLOCK_N"]))
    _moe_gemv_batched_kernel[grid](x, w, s, o, K, N, Kp, s.shape[2], sxe,
                                   BLOCK_N=64, GROUP=group_size, BIAS=_BIAS)
    return o


def _moe_decode(method, layer, x, topk_weights, topk_ids):
    G = method.group_size
    w13p, w13s = layer.w13_weight_packed, layer.w13_weight_scale
    w2p, w2s = layer.w2_weight_packed, layer.w2_weight_scale
    apply_on_input = getattr(layer, "apply_router_weight_on_input", False)
    x0 = x[0]                                    # [H] shared across the active experts at decode
    # Gather the active experts ONCE via index_select (cudagraph-capturable); indexing a weight
    # tensor with a GPU scalar would throw hipErrorStreamCaptureUnsupported during graph capture.
    ids = topk_ids[0]
    w13pe = w13p.index_select(0, ids); w13se = w13s.index_select(0, ids)
    w2pe = w2p.index_select(0, ids); w2se = w2s.index_select(0, ids)
    wts = topk_weights[0].to(x.dtype)            # [tk]
    if _HIP_MOE:
        # Native HIP fused MoE-decode kernel (warp-per-row, dwordx4, ~454 GB/s).
        y = torch.ops.vllm_win_moe.moe_decode_w4(
            x0.contiguous(), w13pe.contiguous(), w13se.contiguous(),
            w2pe.contiguous(), w2se.contiguous(), wts.contiguous(), int(G), 8)
        return y.unsqueeze(0)
    # gate_up for ALL active experts in ONE launch (x shared) -> [tk, 2*I]
    gate_up = _moe_gemv_batched(x0, w13pe, w13se, G, x_per_expert=False)
    half = gate_up.shape[1] // 2
    act = torch.nn.functional.gelu(
        gate_up[:, :half], approximate="tanh") * gate_up[:, half:]  # GeGLU (gemma) [tk, I]
    act = act.contiguous()
    # down for ALL active experts in ONE launch (x per-expert) -> [tk, H]
    down = _moe_gemv_batched(act, w2pe, w2se, G, x_per_expert=True)
    y = (wts[:, None] * down).sum(0)             # weighted sum over the top_k experts -> [H]
    return y.unsqueeze(0)


_HIP_MOE = False


def _load_hip_moe():
    global _HIP_MOE
    if os.environ.get("VLLM_WIN_MOE_HIP", "0") != "1":
        return
    import glob
    d = os.environ.get("VLLM_WIN_MOE_HIP_DIR", r"C:\vw_moedev_build")
    for p in sorted(glob.glob(os.path.join(d, "*.pyd"))):
        try:
            torch.ops.load_library(p)
            if hasattr(torch.ops, "vllm_win_moe") and hasattr(torch.ops.vllm_win_moe, "moe_decode_w4"):
                _HIP_MOE = True
                print("vllm-win: loaded native HIP MoE-decode kernel from", p)
                return
        except Exception as e:  # noqa: BLE001
            print("vllm-win HIP MoE load warning:", repr(e))


_PATCHED = False


def patch_moe() -> None:
    global _PATCHED
    if _PATCHED or os.environ.get("VLLM_WIN_MOE_DECODE", "0") != "1":
        return
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            CompressedTensorsWNA16MoEMethod,
        )
    except Exception as e:  # noqa: BLE001
        print("vllm-win moe_decode patch warning:", repr(e))
        return
    _load_hip_moe()
    validate = os.environ.get("VLLM_WIN_MOE_VALIDATE", "0") == "1"
    _orig = CompressedTensorsWNA16MoEMethod.apply

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        if x.shape[0] == 1 and not validate:
            return _moe_decode(self, layer, x, topk_weights, topk_ids)
        out = _orig(self, layer, x, topk_weights, topk_ids, shared_experts_input)
        if validate and x.shape[0] == 1:
            try:
                mine = _moe_decode(self, layer, x, topk_weights, topk_ids)
                ref = out[0] if isinstance(out, tuple) else out
                rel = (mine.float() - ref.float()).abs().max().item() / (
                    ref.float().abs().max().item() + 1e-6)
                print(f"MOE_DECODE_VALIDATE rel={rel:.4e}")
            except Exception as e:  # noqa: BLE001
                print("MOE_DECODE_VALIDATE err:", repr(e))
        return out

    CompressedTensorsWNA16MoEMethod.apply = apply
    _PATCHED = True
    print("vllm-win: patched CompressedTensorsWNA16MoEMethod.apply (M=1 MoE decode GEMV)")
